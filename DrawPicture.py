import sys
import json
import os
import matplotlib.pyplot as plt
from datetime import datetime


def save_training_data(filename, **kwargs):
    """
    Save training data + agent attributes to JSON file.
    """
    filename += datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".json"
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)

    with open(filename, "w") as f:
        try:
            json.dump(kwargs, f, indent=4, default=str)  # convert non-serializable to str
        except TypeError:
            pass
    print(f"Data saved to {filename} ({', '.join(kwargs.keys())})")


def plot_training_data(filename):
    """
    Plot only numeric list data from JSON file (ignore static attributes).
    """
    with open(filename, "r") as f:
        data = json.load(f)

    plt.figure(figsize=(8, 5))

    for key, values in data.items():
        # Only plot lists of numbers
        if isinstance(values, list) and all(isinstance(v, (int, float)) for v in values):
            epochs = range(1, len(values) + 1)
            plt.plot(epochs, values, label=key)

    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training Metrics")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 DrawPicture.py record/[filename].json")
        sys.exit(0)
    plot_training_data(sys.argv[1])
