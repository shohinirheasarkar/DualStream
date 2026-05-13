"""
run_dualstream.py

One-command driver for the working DualStream prototype.

Run:
    PYTHONPATH=. python scripts/run_dualstream.py
"""

from pathlib import Path

from src.training.train import train_dualstream


def main():
    synthetic_dir = Path("data/synthetic/example_001")

    if not (synthetic_dir / "observed_dff.npy").exists():
        raise FileNotFoundError(
            "Synthetic data not found. Run: python src/data/synthetic.py"
        )

    train_dualstream(
        observed_dff_path="data/synthetic/example_001/observed_dff.npy",
        true_graph_path="data/synthetic/example_001/true_graph.npy",
        output_dir="outputs/dualstream_prototype",
        seed=42,
        window_size=200,
        stride=100,
        epochs=100,
        patience=15,
        lr=1e-3,
    )


if __name__ == "__main__":
    main()