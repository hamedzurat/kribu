"""Simple inference test script for Sholo Guti PyTorch AI."""

from inference.player import NeuralPlayer


def main():
    print("Initializing NeuralPlayer...")
    try:
        player = NeuralPlayer()
    except FileNotFoundError as e:
        print(e)
        return

    # Test with initial board state (me = bits 21-36, opp = bits 0-15)
    # me = 0x0000'001F'FFE0'0000 = 137436856320
    # opp = 0x0000'0000'0000'FFFF = 65535
    print("Testing initial board state...")

    best_move_idx, win_prob = player.get_move(me_mask=137436856320, opp_mask=65535, active_capture_idx=-1)

    print("--- Inference Result ---")
    print(f"Predicted Best Move ID: {best_move_idx}")
    print(f"Predicted Win Probability: {win_prob:.4f}")


if __name__ == "__main__":
    main()
