import os
import torch
import torch.nn as nn
from rich.live import Live
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.layout import Layout

from .config import config
from .dataset import get_dataloaders
from .model import SholoGutiNet


def infinite_iter(dataloader):
    while True:
        for batch in dataloader:
            yield batch


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Check for CUDA architecture mismatch (e.g. GTX 1050 Ti vs modern PyTorch)
    if device.type == "cuda":
        try:
            # Dummy operation to trigger CUBLAS_STATUS_ARCH_MISMATCH if it exists
            _ = torch.nn.Linear(1, 1).to(device)(torch.zeros(1, 1).to(device))
        except RuntimeError:
            device = torch.device("cpu")

    # Initialize model
    model = SholoGutiNet(
        input_features=config.input_features,
        hidden_dim=config.hidden_dim,
        num_residual_blocks=config.num_residual_blocks,
        action_space=config.action_space,
    ).to(device)

    # Optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()

    # Loaders
    policy_loader, value_loader = get_dataloaders(config)
    policy_iter = infinite_iter(policy_loader)
    value_iter = infinite_iter(value_loader)

    os.makedirs(config.save_dir, exist_ok=True)

    from rich import box
    from rich.columns import Columns
    from rich.text import Text
    import shutil

    # Two separate simple progress bars to place side-by-side
    progress_epoch = Progress(
        TextColumn("[green]Epochs:"), BarColumn(), TextColumn("[progress.percentage]{task.percentage:>3.0f}%")
    )
    progress_step = Progress(
        TextColumn("[cyan]Steps:"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
    )

    epoch_task = progress_epoch.add_task("", total=config.epochs)
    step_task = progress_step.add_task("", total=1000)
    steps_per_epoch = 1000

    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(0)
    else:
        device_name = "CPU"

    layout = Layout()
    layout.split_column(Layout(name="header", size=1), Layout(name="body"), Layout(name="footer", size=1))

    layout["footer"].update(Columns([progress_epoch, progress_step], expand=True))

    history_data = []
    best_loss = float("inf")

    def generate_header(cur_ep, cur_pol, cur_val, best):
        b_str = f"{best:.4f}" if best != float("inf") else "---"
        p_str = f"{cur_pol:.4f}" if cur_pol else "---"
        v_str = f"{cur_val:.4f}" if cur_val else "---"
        text = f"Device: {device_name}  |  Epoch: {cur_ep}/{config.epochs}  |  LR: {config.learning_rate:.1e}  |  Batch: {config.batch_size}  |  Best Loss: {b_str}  |  Pol: {p_str}  |  Val: {v_str}"
        return Text(text, style="bold cyan", justify="center")

    def generate_body(hist):
        terminal_height = shutil.get_terminal_size().lines
        terminal_width = shutil.get_terminal_size().columns

        # header(1) + footer(1) + panel_borders(2) = 4 lines overhead
        max_rows = max(5, terminal_height - 6)

        chunks = [hist[i : i + max_rows] for i in range(0, len(hist), max_rows)]

        # Calculate how many columns can fit (each table is ~30-35 chars wide)
        max_columns = max(1, (terminal_width - 4) // 35)
        if len(chunks) > max_columns:
            chunks = chunks[-max_columns:]

        tables = []
        for chunk in chunks:
            t = Table(box=box.SIMPLE)
            t.add_column("Ep", style="cyan")
            t.add_column("Pol", style="magenta")
            t.add_column("Val", style="green")
            t.add_column("Tot", style="yellow")
            for ep, pol, val, tot in chunk:
                t.add_row(str(ep), f"{pol:.4f}", f"{val:.4f}", f"{tot:.4f}")
            tables.append(t)

        return Panel(Columns(tables), title="[b]Metrics History[/b]", box=box.ROUNDED)

    layout["header"].update(generate_header(0, 0.0, 0.0, best_loss))
    layout["body"].update(generate_body(history_data))

    with Live(layout, refresh_per_second=4):
        for epoch in range(1, config.epochs + 1):
            model.train()
            total_pol_loss = 0.0
            total_val_loss = 0.0

            progress_step.reset(step_task)
            for step in range(steps_per_epoch):
                p_x, p_target, _ = next(policy_iter)
                v_x, _, v_target = next(value_iter)

                p_x = p_x.to(device)
                p_target = p_target.to(device)

                v_x = v_x.to(device)
                v_target = v_target.to(device)

                optimizer.zero_grad()

                p_logits, _ = model(p_x)
                loss_p = policy_criterion(p_logits, p_target)

                _, v_pred = model(v_x)
                loss_v = value_criterion(v_pred, v_target)

                loss = loss_p + config.value_loss_weight * loss_v
                loss.backward()
                optimizer.step()

                total_pol_loss += loss_p.item()
                total_val_loss += loss_v.item()

                progress_step.advance(step_task)

            avg_pol = total_pol_loss / steps_per_epoch
            avg_val = total_val_loss / steps_per_epoch
            avg_tot = avg_pol + config.value_loss_weight * avg_val

            history_data.append((epoch, avg_pol, avg_val, avg_tot))

            if avg_tot < best_loss:
                best_loss = avg_tot
                torch.save(model.state_dict(), os.path.join(config.save_dir, "best_model.pt"))

            if epoch % config.save_every_epochs == 0:
                torch.save(model.state_dict(), os.path.join(config.save_dir, f"model_ep{epoch}.pt"))

            layout["header"].update(generate_header(epoch, avg_pol, avg_val, best_loss))
            layout["body"].update(generate_body(history_data))
            progress_epoch.advance(epoch_task)
