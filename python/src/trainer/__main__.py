"""Entry point for the Sholo Guti PyTorch trainer."""

import sys


def main():
    try:
        from .train import train

        train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError during training: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
