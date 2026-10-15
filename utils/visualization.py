"""
ELEC 475: Lab 2 - Plotting & Visualization Utilities
"""

import os
import argparse
import matplotlib.pyplot as plt
from typing import List, Dict


# ============================================================
# INDIVIDUAL TRAINING CURVES
# ============================================================

def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    save_path: str = "./plots/training_curves.png",
    model_name: str = "Model"
):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-o', label='Train Loss')
    plt.plot(epochs, val_losses, 'r--s', label='Val Loss')
    plt.title(f"{model_name.upper()} - Cross-Entropy Loss", fontsize=12)
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("Loss", fontsize=10)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)

    # Accuracy Curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'b-o', label='Train Accuracy')
    plt.plot(epochs, val_accs, 'g--^', label='Val Accuracy')
    plt.title(f"{model_name.upper()} - Top-1 Accuracy (%)", fontsize=12)
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("Accuracy (%)", fontsize=10)
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"Training curves successfully saved to: {save_path}")


# ============================================================
# MASTER ABLATION SUMMARY
# ============================================================

def plot_ablation_summary(
    cifar_results: Dict[str, Dict[str, List[float]]],
    tiny_results: Dict[str, Dict[str, List[float]]] = None,
    save_path: str = "./plots/ablation_summary.png"
):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    colors = [
        '#1f77b4',
        '#ff7f0e',
        '#2ca02c',
        '#d62728',
        '#9467bd',
        '#8c564b',
        '#e377c2'
    ]

    styles = [
        '-',
        '--',
        '-.',
        ':',
        '-',
        '--',
        '-.'
    ]

    # If only CIFAR-10 data is supplied, preserve original behaviour
    if tiny_results is None:

        plt.figure(figsize=(10, 6))

        for idx, (model_label, data) in enumerate(cifar_results.items()):

            val_accs = data.get("val_accs", [])
            epochs = range(1, len(val_accs) + 1)

            plt.plot(
                epochs,
                val_accs,
                linestyle=styles[idx % len(styles)],
                marker='o',
                color=colors[idx % len(colors)],
                label=model_label,
                linewidth=2
            )

        plt.title(
            "ELEC 475: Master Ablation Study (Validation Accuracy)",
            fontsize=13
        )

        plt.xlabel("Epoch", fontsize=11)
        plt.ylabel("Validation Accuracy (%)", fontsize=11)

        plt.legend(loc='lower right', frameon=True)
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()

        print(f"Master ablation summary plot saved to: {save_path}")
        return

    # --------------------------------------------------------
    # CIFAR-10 + Tiny ImageNet side-by-side
    # --------------------------------------------------------

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # CIFAR-10
    ax = axes[0]

    for idx, (model_label, data) in enumerate(cifar_results.items()):

        val_accs = data.get("val_accs", [])
        epochs = range(1, len(val_accs) + 1)

        ax.plot(
            epochs,
            val_accs,
            linestyle=styles[idx % len(styles)],
            marker='o',
            color=colors[idx % len(colors)],
            label=model_label,
            linewidth=2
        )

    ax.set_title(
        "CIFAR-10 Validation Accuracy",
        fontsize=13
    )

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Validation Accuracy (%)", fontsize=11)

    ax.grid(
        True,
        linestyle='--',
        alpha=0.7
    )

    ax.legend(
        loc='lower right',
        frameon=True,
        fontsize=8
    )

    # Tiny ImageNet
    ax = axes[1]

    for idx, (model_label, data) in enumerate(tiny_results.items()):

        val_accs = data.get("val_accs", [])
        epochs = range(1, len(val_accs) + 1)

        ax.plot(
            epochs,
            val_accs,
            linestyle=styles[idx % len(styles)],
            marker='o',
            color=colors[idx % len(colors)],
            label=model_label,
            linewidth=2
        )

    ax.set_title(
        "Tiny ImageNet Validation Accuracy",
        fontsize=13
    )

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Validation Accuracy (%)", fontsize=11)

    ax.grid(
        True,
        linestyle='--',
        alpha=0.7
    )

    ax.legend(
        loc='upper left',
        frameon=True,
        fontsize=8
    )

    fig.suptitle(
        "ELEC 475: Master Ablation Study",
        fontsize=15
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.savefig(
        save_path,
        dpi=200
    )

    plt.close()

    print(f"Master ablation summary plot saved to: {save_path}")


# ============================================================
# CIFAR-10 MASTER ABLATION DATA
# ============================================================

CIFAR10_ABLATION_RESULTS = {

    "Base LeNet-5": {
        "val_accs": [
            46.67,
            50.25,
            50.88,
            53.73,
            54.18
        ]
    },

    "Modernized LeNet": {
        "val_accs": [
            53.42,
            59.33,
            61.74,
            64.71,
            65.46
        ]
    },

    "AlexNet (No Norm)": {
        "val_accs": [
            44.86,
            56.82,
            61.41,
            65.21,
            68.52,
            68.48,
            72.56,
            71.83,
            73.30,
            73.67
        ]
    },

    "AlexNet + LRN": {
        "val_accs": [
            45.69,
            57.23,
            63.82,
            68.10,
            70.98,
            70.83,
            72.89,
            73.51,
            74.95,
            76.26
        ]
    },

    "AlexNet + BatchNorm": {
        "val_accs": [
            51.66,
            65.69,
            66.64,
            69.05,
            72.07,
            70.59,
            77.75,
            75.87,
            78.93,
            78.11
        ]
    },

    "AlexNet + BN + Aug": {
        "val_accs": [
            52.21,
            60.55,
            66.23,
            67.91,
            70.55,
            71.84,
            76.83,
            77.30,
            80.27,
            79.92,
            80.00,
            74.20,
            79.84,
            81.01,
            83.26
        ]
    },

    "PyTorch Zoo AlexNet": {
        "val_accs": [
            39.36,
            49.08,
            55.42,
            58.42,
            59.41,
            60.09,
            62.65,
            63.94,
            64.14,
            64.31,
            65.15,
            65.49,
            66.58,
            65.42,
            66.94
        ]
    }
}


# ============================================================
# TINY IMAGENET MASTER ABLATION DATA
# ============================================================

TINY_IMAGENET_ABLATION_RESULTS = {

    "Base LeNet-5": {
        "val_accs": [
            7.38,
            9.08,
            10.34,
            11.61,
            12.90
        ]
    },

    "Modernized LeNet": {
        "val_accs": [
            10.77,
            15.93,
            18.70,
            20.92,
            21.95
        ]
    },

    "AlexNet (No Norm)": {
        "val_accs": [
            2.85,
            5.80,
            8.41,
            10.72,
            12.30,
            12.87,
            14.75,
            15.97,
            16.82,
            17.41
        ]
    },

    "AlexNet + LRN": {
        "val_accs": [
            1.85,
            2.97,
            5.02,
            6.97,
            8.15,
            9.86,
            11.00,
            12.27,
            12.99,
            13.85
        ]
    },

    "AlexNet + BatchNorm": {
        "val_accs": [
            7.85,
            12.57,
            16.23,
            20.62,
            23.27,
            23.82,
            28.97,
            31.36,
            34.80,
            37.18
        ]
    },

    "AlexNet + BN + Aug": {
        "val_accs": [
            6.80,
            9.97,
            13.38,
            19.41,
            18.22,
            22.26,
            26.23,
            28.46,
            27.16,
            33.12
        ]
    },

    "PyTorch Zoo AlexNet": {
    "val_accs": [
        1.70,
        3.15,
        5.25,
        7.42,
        8.82,
        9.94,
        10.32,
        11.41,
        12.68,
        12.69,
        13.65,
        14.17,
        14.08,
        14.60,
        15.45
        ]
    }
}


# ============================================================
# COMMAND-LINE INTERFACE
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="ELEC 475: Lab 2 Visualization Utilities"
    )

    parser.add_argument(
        "--plot-ablation",
        action="store_true",
        help="Generate the CIFAR-10 and Tiny ImageNet master ablation plot"
    )

    args = parser.parse_args()

    if args.plot_ablation:

        plot_ablation_summary(
            CIFAR10_ABLATION_RESULTS,
            TINY_IMAGENET_ABLATION_RESULTS,
            save_path="./plots/ablation_summary.png"
        )