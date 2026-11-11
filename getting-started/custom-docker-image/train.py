"""Verify River packaging by training a CPU classifier on synthetic data.

Running this module prints one JSON record with the library version, sample
counts, and held-out accuracy. It needs no downloaded data or model artifacts.
"""

import json
import random
from collections.abc import Iterator
from importlib.metadata import version

from river import linear_model, metrics, preprocessing


def examples(count: int, seed: int) -> Iterator[tuple[dict[str, float], int]]:
    """Yield reproducible two-feature samples with a linear decision boundary.

    Args:
        count: Number of examples to generate.
        seed: Seed for an isolated random generator; global RNG state is unchanged.

    Yields:
        A feature dictionary and a binary label indicating whether x1 + x2 > 0.
    """
    rng = random.Random(seed)
    for _ in range(count):
        x1 = rng.gauss(0, 1)
        x2 = rng.gauss(0, 1)
        yield {"x1": x1, "x2": x2}, int(x1 + x2 > 0)


def main() -> None:
    """Train on 400 samples, evaluate on 100 fresh samples, and print JSON."""
    model = preprocessing.StandardScaler() | linear_model.LogisticRegression()
    for features, label in examples(400, seed=7):
        model.learn_one(features, label)

    # A different seed keeps evaluation samples separate from the training stream.
    accuracy = metrics.Accuracy()
    for features, label in examples(100, seed=8):
        accuracy.update(label, model.predict_one(features))

    # Flush the result immediately so it appears promptly in AIR's streamed logs.
    print(
        json.dumps(
            {
                "library": "river",
                "version": version("river"),
                "train_examples": 400,
                "test_examples": 100,
                "test_accuracy": round(accuracy.get(), 3),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
