from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlx_tune.arithmetic_grpo_validation import main


if __name__ == "__main__":
    raise SystemExit(main())
