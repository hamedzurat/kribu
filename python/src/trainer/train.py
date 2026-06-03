import math
import os
import shutil

from rich.columns import Columns
import torch
import torch.nn as nn
from rich.layout import Layout
from rich.live import Live
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
from rich.text import Text
from torch.utils.tensorboard import SummaryWriter

from .config import config
from .dataset import get_dataloaders
from .model import SholoGutiNet


def infinite_iter(dataloader):
    """Yield batches from a dataloader forever, restarting when it is exhausted."""
    while True:
        for batch in dataloader:
            yield batch


def resolve_steps_per_epoch(policy_batches: int, value_batches: int | None, requested_steps: int | None) -> int:
    """Return the number of optimizer updates to run for each epoch."""
    if requested_steps is not None:
        if requested_steps <= 0:
            raise ValueError("steps_per_epoch must be positive when set")
        return requested_steps

    if policy_batches <= 0:
        raise ValueError("policy dataloader must not be empty")
    if value_batches is None:
        return policy_batches
    if value_batches <= 0:
        raise ValueError("value dataloader must not be empty when provided")
    return max(policy_batches, value_batches)


def dataset_passes(steps_per_epoch: int, batches_per_dataset: int) -> float:
    """Return how many dataloader passes an epoch performs for one dataset."""
    if batches_per_dataset <= 0:
        raise ValueError("batches_per_dataset must be positive")
    return steps_per_epoch / batches_per_dataset


def current_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    """Return the learning rate from the optimizer's first parameter group."""
    return optimizer.param_groups[0]["lr"]


def loss_total(policy_loss: float, value_loss: float, value_loss_weight: float) -> float:
    """Return the weighted combined trainer loss."""
    if value_loss_weight == 0.0 or not is_finite_metric(value_loss):
        return policy_loss
    return policy_loss + value_loss_weight * value_loss


def is_improved(metric: float, best_metric: float, min_improvement: float) -> bool:
    """Return true when a metric improves by at least the configured margin."""
    return metric < best_metric - min_improvement


def is_finite_metric(value: float) -> bool:
    """Return true when a metric can be plotted."""
    return isinstance(value, int | float) and math.isfinite(float(value))


def metric_points(history: list[tuple], metric_index: int) -> list[tuple[int, float]]:
    """Return finite epoch/value points from trainer history."""
    points = []
    for row in history:
        if len(row) > metric_index and is_finite_metric(row[metric_index]):
            points.append((int(row[0]), float(row[metric_index])))
    return points


def compact_label(text: str, width: int) -> str:
    """Trim a label so it fits in a fixed-width terminal area."""
    if len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    return text[: width - 3] + "..."


def render_line_chart(
    history: list[tuple],
    series: list[tuple[str, int, str]],
    title: str,
    width: int,
    height: int,
) -> list[str]:
    """Render one ASCII line chart using as much of the requested area as possible."""
    width = max(24, width)
    height = max(4, height)
    label_width = 9
    plot_width = max(10, width - label_width - 2)
    plot_height = max(2, height - 2)

    series_points = [(name, marker, metric_points(history, metric_index)) for name, metric_index, marker in series]
    all_points = [point for _, _, points in series_points for point in points]
    if not all_points:
        return [compact_label(f"{title}: waiting for metrics", width)] + [" " * width for _ in range(height - 1)]

    x_min = min(epoch for epoch, _ in all_points)
    x_max = max(epoch for epoch, _ in all_points)
    y_min = min(value for _, value in all_points)
    y_max = max(value for _, value in all_points)
    if y_min == y_max:
        padding = max(abs(y_min) * 0.05, 1e-6)
        y_min -= padding
        y_max += padding

    canvas = [[" " for _ in range(plot_width)] for _ in range(plot_height)]
    x_span = max(1, x_max - x_min)
    y_span = y_max - y_min

    for _, marker, points in series_points:
        for epoch, value in points:
            x = round((epoch - x_min) / x_span * (plot_width - 1))
            y = plot_height - 1 - round((value - y_min) / y_span * (plot_height - 1))
            existing = canvas[y][x]
            canvas[y][x] = marker if existing in {" ", marker} else "#"

    legend = " ".join(f"{marker}={name}" for name, _, marker in series)
    lines = [compact_label(f"{title}  {legend}", width)]
    for row_idx, row in enumerate(canvas):
        y_value = y_max - (row_idx / max(1, plot_height - 1)) * y_span
        lines.append(compact_label(f"{y_value:>8.4g} |{''.join(row)}", width))

    end_label = str(x_max)
    axis_space = max(0, plot_width - len(str(x_min)) - len(end_label))
    axis = f"{'epoch':>{label_width}}  {x_min}{' ' * axis_space}{end_label}"
    lines.append(compact_label(axis, width))
    return lines[:height]


def render_training_dashboard(history: list[tuple], width: int, height: int) -> Text:
    """Render full-screen trainer history graphs."""
    width = max(24, width)
    height = max(6, height)
    if not history:
        return Text("Waiting for first completed epoch...", style="dim")

    sections: list[list[str]] = []
    if height >= 24:
        loss_height = max(8, height // 2)
        remaining = height - loss_height
        acc_height = max(6, remaining // 2)
        mae_height = max(4, height - loss_height - acc_height)
        sections.append(
            render_line_chart(history, [("train", 1, "*"), ("validation", 2, "o")], "Total loss", width, loss_height)
        )
        sections.append(render_line_chart(history, [("policy accuracy", 3, "+")], "Policy accuracy", width, acc_height))
        sections.append(render_line_chart(history, [("value MAE", 4, "x")], "Value MAE", width, mae_height))
    elif height >= 14:
        loss_height = max(7, height - 6)
        sections.append(
            render_line_chart(history, [("train", 1, "*"), ("validation", 2, "o")], "Total loss", width, loss_height)
        )
        sections.append(
            render_line_chart(
                history,
                [("policy accuracy", 3, "+"), ("value MAE", 4, "x")],
                "Accuracy / MAE",
                width,
                height - loss_height,
            )
        )
    else:
        sections.append(
            render_line_chart(history, [("train", 1, "*"), ("validation", 2, "o")], "Total loss", width, height)
        )

    lines = [line for section in sections for line in section]
    return Text("\n".join(lines[:height]))


def checkpoint_path() -> str:
    """Return the full path to the resumable checkpoint."""
    return os.path.join(config.save_dir, "last_checkpoint.pt")


def load_checkpoint(model, optimizer, scheduler, scaler, device):
    """Load a resumable training checkpoint when configured and available."""
    path = checkpoint_path()
    if not config.resume or not os.path.exists(path):
        return 1, float("inf"), 0, []

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    if "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])

    return (
        checkpoint["epoch"] + 1,
        checkpoint.get("best_validation_loss", float("inf")),
        checkpoint.get("checks_without_improvement", 0),
        checkpoint.get("history", []),
    )


def save_checkpoint(model, optimizer, scheduler, scaler, epoch: int, best_validation_loss: float, checks: int, history):
    """Save a resumable training checkpoint."""
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_validation_loss": best_validation_loss,
            "checks_without_improvement": checks,
            "history": history,
        },
        checkpoint_path(),
    )


def weighted_mean(losses: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Return a numerically stable weighted mean for per-sample losses."""
    sample_weights = weights.to(device=losses.device, dtype=losses.dtype)
    weight_sum = sample_weights.sum().clamp_min(torch.finfo(losses.dtype).eps)
    return (losses * sample_weights).sum() / weight_sum


def apply_policy_mask(
    policy_logits: torch.Tensor,
    legal_mask: torch.Tensor | None,
    policy_target: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mask policy logits so loss and accuracy only compare legal actions."""
    if legal_mask is None:
        return policy_logits

    mask = legal_mask.to(device=policy_logits.device, dtype=torch.bool)
    if policy_target is not None:
        mask = mask.clone()
        mask.scatter_(1, policy_target[:, None], True)
    return policy_logits.masked_fill(~mask, torch.finfo(policy_logits.dtype).min)


def move_batch_to_device(batch, device):
    """Move a trainer batch to the selected device."""
    features, policy_target, value_target, legal_mask, policy_weight, value_weight = batch
    if legal_mask is not None:
        legal_mask = legal_mask.to(device)
    return (
        features.to(device),
        policy_target.to(device),
        value_target.to(device),
        legal_mask,
        policy_weight.to(device),
        value_weight.to(device),
    )


@torch.no_grad()
def evaluate(
    model, policy_loader, value_loader, policy_criterion, value_criterion, device
) -> tuple[float, float, float, float]:
    """Evaluate validation policy/value losses, policy accuracy, and value MAE."""
    if policy_loader is None:
        return float("nan"), float("nan"), float("nan"), float("nan")

    model.eval()
    total_policy_loss = 0.0
    correct_policy = 0
    policy_samples = 0
    for p_x, p_target, _, p_legal_mask, p_weight, _ in policy_loader:
        p_x = p_x.to(device)
        p_target = p_target.to(device)
        p_weight = p_weight.to(device)
        if p_legal_mask is not None:
            p_legal_mask = p_legal_mask.to(device)
        p_logits, _ = model(p_x)
        p_logits = apply_policy_mask(p_logits, p_legal_mask, p_target)
        total_policy_loss += weighted_mean(policy_criterion(p_logits, p_target), p_weight).item()
        correct_policy += int((p_logits.argmax(dim=-1) == p_target).sum().item())
        policy_samples += p_target.numel()

    policy_loss = total_policy_loss / len(policy_loader)
    policy_accuracy = correct_policy / policy_samples

    if value_loader is None:
        return policy_loss, float("nan"), policy_accuracy, float("nan")

    total_value_loss = 0.0
    total_value_abs_error = 0.0
    value_samples = 0
    for v_x, _, v_target, _, _, v_weight in value_loader:
        v_x = v_x.to(device)
        v_target = v_target.to(device)
        v_weight = v_weight.to(device)
        _, v_pred = model(v_x)
        total_value_loss += weighted_mean(value_criterion(v_pred, v_target), v_weight).item()
        total_value_abs_error += float((torch.abs(v_pred - v_target) * v_weight).sum().item())
        value_samples += float(v_weight.sum().item())

    value_loss = total_value_loss / len(value_loader)
    value_mae = total_value_abs_error / value_samples
    return policy_loss, value_loss, policy_accuracy, value_mae


def train():
    """Train the Sholo Guti policy/value network."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        try:
            _ = torch.nn.Linear(1, 1).to(device)(torch.zeros(1, 1).to(device))
        except RuntimeError:
            device = torch.device("cpu")

    model = SholoGutiNet(
        input_features=config.input_features,
        hidden_dim=config.hidden_dim,
        num_residual_blocks=config.num_residual_blocks,
        action_space=config.action_space,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=config.lr_plateau_factor,
        patience=config.lr_plateau_patience,
        min_lr=config.min_learning_rate,
    )
    policy_criterion = nn.CrossEntropyLoss(reduction="none")
    value_criterion = nn.MSELoss(reduction="none")
    use_mixed_precision = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_mixed_precision)

    policy_loader, value_loader, policy_validation_loader, value_validation_loader = get_dataloaders(config)
    effective_value_loss_weight = 0.0 if config.policy_only else config.value_loss_weight
    if config.policy_only:
        value_loader = None
        value_validation_loader = None
    policy_batches = len(policy_loader)
    value_batches = None if value_loader is None else len(value_loader)
    steps_per_epoch = resolve_steps_per_epoch(policy_batches, value_batches, config.steps_per_epoch)
    policy_passes = dataset_passes(steps_per_epoch, policy_batches)
    value_passes = float("nan") if value_batches is None else dataset_passes(steps_per_epoch, value_batches)
    policy_iter = infinite_iter(policy_loader)
    value_iter = None if value_loader is None else infinite_iter(value_loader)

    os.makedirs(config.save_dir, exist_ok=True)

    start_epoch, best_validation_loss, checks_without_improvement, history_data = load_checkpoint(
        model, optimizer, scheduler, scaler, device
    )

    tensorboard_dir = os.path.join(config.save_dir, "tensorboard", "current")
    writer = SummaryWriter(log_dir=tensorboard_dir, purge_step=max(0, start_epoch - 1))

    progress_epoch = Progress(
        TextColumn("[green]Epochs:"), BarColumn(), TextColumn("[progress.percentage]{task.percentage:>3.0f}%")
    )
    progress_step = Progress(
        TextColumn("[cyan]Steps:"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
    )

    epoch_task = progress_epoch.add_task("", total=config.epochs, completed=max(0, start_epoch - 1))
    step_task = progress_step.add_task("", total=steps_per_epoch)
    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"

    if start_epoch == 1:
        writer.add_text("run/device", device.type)
        writer.add_text("run/device_name", device_name)
        writer.add_text("run/save_dir", config.save_dir)

        writer.add_scalar("config/steps_per_epoch", steps_per_epoch, 0)
        writer.add_scalar("config/policy_passes_per_epoch", policy_passes, 0)
        if is_finite_metric(value_passes):
            writer.add_scalar("config/value_passes_per_epoch", value_passes, 0)
        writer.add_scalar("config/batch_size", config.batch_size, 0)
        writer.add_scalar("config/value_loss_weight", effective_value_loss_weight, 0)
        writer.add_scalar("config/policy_only", 1 if config.policy_only else 0, 0)

    layout = Layout()
    layout.split_column(Layout(name="header", size=1), Layout(name="body"), Layout(name="footer", size=1))
    layout["footer"].update(Columns([progress_epoch, progress_step], expand=True))

    def generate_header(cur_ep, cur_pol, cur_val, cur_validation_loss, best):
        terminal_width = shutil.get_terminal_size().columns
        b_str = f"{best:.4f}" if best != float("inf") else "---"
        p_str = f"{cur_pol:.4f}" if cur_pol else "---"
        v_str = f"{cur_val:.4f}" if is_finite_metric(cur_val) else "---"
        val_str = f"{cur_validation_loss:.4f}" if cur_validation_loss == cur_validation_loss else "---"
        value_pass_label = f"{value_passes:.2f}x" if is_finite_metric(value_passes) else "---"
        text = (
            f"Device: {device_name}  |  Epoch: {cur_ep}/{config.epochs}  |  Steps: {steps_per_epoch}  |  "
            f"Passes: P {policy_passes:.2f}x / V {value_pass_label}  |  LR: {current_learning_rate(optimizer):.1e}  |  "
            f"Batch: {config.batch_size}  |  Best Val: {b_str}  |  Train: P {p_str} / V {v_str}  |  Val: {val_str}"
        )
        return Text(compact_label(text, terminal_width), style="bold cyan", justify="center")

    def generate_body(hist):
        terminal_size = shutil.get_terminal_size()
        return render_training_dashboard(hist, terminal_size.columns, max(6, terminal_size.lines - 2))

    layout["header"].update(generate_header(start_epoch - 1, 0.0, 0.0, float("nan"), best_validation_loss))
    layout["body"].update(generate_body(history_data))

    try:
        with Live(layout, refresh_per_second=4):
            for epoch in range(start_epoch, config.epochs + 1):
                model.train()
                total_pol_loss = 0.0
                total_val_loss = 0.0

                progress_step.reset(step_task)
                for _ in range(steps_per_epoch):
                    p_x, p_target, _, p_legal_mask, p_weight, _ = move_batch_to_device(next(policy_iter), device)

                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device.type, enabled=use_mixed_precision):
                        p_logits, _ = model(p_x)
                        p_logits = apply_policy_mask(p_logits, p_legal_mask, p_target)
                        loss_p = weighted_mean(policy_criterion(p_logits, p_target), p_weight)
                        if value_iter is None:
                            loss_v = torch.zeros((), device=device)
                        else:
                            v_x, _, v_target, _, _, v_weight = move_batch_to_device(next(value_iter), device)
                            _, v_pred = model(v_x)
                            loss_v = weighted_mean(value_criterion(v_pred, v_target), v_weight)
                        loss = loss_p + effective_value_loss_weight * loss_v

                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()

                    total_pol_loss += loss_p.item()
                    total_val_loss += loss_v.item()
                    progress_step.advance(step_task)

                avg_pol = total_pol_loss / steps_per_epoch
                avg_val = float("nan") if value_iter is None else total_val_loss / steps_per_epoch
                avg_tot = loss_total(avg_pol, avg_val, effective_value_loss_weight)
                validation_total = float("nan")
                policy_accuracy = float("nan")
                value_mae = float("nan")

                if epoch % config.validate_every_epochs == 0:
                    val_pol, val_value, policy_accuracy, value_mae = evaluate(
                        model,
                        policy_validation_loader,
                        value_validation_loader,
                        policy_criterion,
                        value_criterion,
                        device,
                    )
                    validation_total = loss_total(val_pol, val_value, effective_value_loss_weight)
                    scheduler.step(validation_total)

                    if is_improved(validation_total, best_validation_loss, config.min_improvement):
                        best_validation_loss = validation_total
                        checks_without_improvement = 0
                        torch.save(model.state_dict(), os.path.join(config.save_dir, "best_model.pt"))
                    else:
                        checks_without_improvement += 1
                else:
                    scheduler.step(avg_tot)

                if policy_validation_loader is None and is_improved(
                    avg_tot, best_validation_loss, config.min_improvement
                ):
                    best_validation_loss = avg_tot
                    torch.save(model.state_dict(), os.path.join(config.save_dir, "best_model.pt"))

                history_data.append((epoch, avg_tot, validation_total, policy_accuracy, value_mae))

                writer.add_scalar("loss/train_total", avg_tot, epoch)
                writer.add_scalar("loss/train_policy", avg_pol, epoch)
                if is_finite_metric(avg_val):
                    writer.add_scalar("loss/train_value", avg_val, epoch)

                if is_finite_metric(validation_total):
                    writer.add_scalar("loss/validation_total", validation_total, epoch)

                if is_finite_metric(policy_accuracy):
                    writer.add_scalar("metrics/policy_accuracy", policy_accuracy, epoch)

                if is_finite_metric(value_mae):
                    writer.add_scalar("metrics/value_mae", value_mae, epoch)

                writer.add_scalar("optimizer/learning_rate", current_learning_rate(optimizer), epoch)

                if is_finite_metric(best_validation_loss):
                    writer.add_scalar("training/best_validation_loss", best_validation_loss, epoch)

                writer.add_scalar("training/checks_without_improvement", checks_without_improvement, epoch)

                if epoch % config.save_every_epochs == 0:
                    torch.save(model.state_dict(), os.path.join(config.save_dir, f"model_ep{epoch}.pt"))

                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    best_validation_loss,
                    checks_without_improvement,
                    history_data,
                )
                writer.flush()

                layout["header"].update(
                    generate_header(epoch, avg_pol, avg_val, validation_total, best_validation_loss)
                )
                layout["body"].update(generate_body(history_data))
                progress_epoch.advance(epoch_task)

                if (
                    policy_validation_loader is not None
                    and config.early_stop_patience > 0
                    and checks_without_improvement >= config.early_stop_patience
                ):
                    break
    finally:
        writer.close()
