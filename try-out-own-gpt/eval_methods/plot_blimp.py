import os
import re
import matplotlib.pyplot as plt

# --- Configuration ---
BASE_DIR = "/local/perdijks/training_gpt/blimp_results"          # Directory containing all checkpoint folders
MODEL_NAMES = ["karpathy_babybabel_without_books", "karpathy_chiscor_sftw"]   # Prefixes for the two models
# ---------------------

def get_checkpoints(base_dir, model_name):
    """
    Scan base_dir for directories matching model_name_<checkpoint>
    and read the BLiMP macro accuracy from the corresponding txt file.
    Returns a sorted list of (checkpoint_step, accuracy) tuples.
    """
    results = []
    pattern = re.compile(rf"^{re.escape(model_name)}_(\d+)$")

    for entry in os.scandir(base_dir):
        if not entry.is_dir():
            continue
        match = pattern.match(entry.name)
        if not match:
            continue

        checkpoint = int(match.group(1))
        txt_file = os.path.join(entry.path, f"{entry.name}_blimp_nl_macro_accuracy.txt")

        if not os.path.isfile(txt_file):
            print(f"  [Warning] Missing file: {txt_file}")
            continue

        with open(txt_file, "r") as f:
            content = f.read().strip()

        try:
            accuracy = float(content)
        except ValueError:
            print(f"  [Warning] Could not parse float from '{content}' in {txt_file}")
            continue

        results.append((checkpoint, accuracy))

    return sorted(results)


def main():
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = ["#1f77b4", "#d62728"]   # Blue, Red — one per model
    markers = ["o", "s"]

    for model_name, color, marker in zip(MODEL_NAMES, colors, markers):
        data = get_checkpoints(BASE_DIR, model_name)

        if not data:
            print(f"[Warning] No data found for {model_name}. Check BASE_DIR and MODEL_NAMES.")
            continue

        steps, accuracies = zip(*data)
        ax.plot(
            steps, accuracies,
            label=model_name,
            color=color,
            marker=marker,
            linewidth=2,
            markersize=6,
        )
        print(f"{model_name}: {len(data)} checkpoints loaded.")

    ax.set_xlabel("Checkpoint", fontsize=12)
    ax.set_ylabel("BLiMP NL macro accuracy", fontsize=12)
    ax.set_title("BLiMP macro accuracy over checkpoints", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    out_path = "blimp_macro_accuracy.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nPlot saved to: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()