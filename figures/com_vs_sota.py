import matplotlib.pyplot as plt
import matplotlib.image as mpimg

compare_sota_b0 = "compare_sota_b0.png"
compare_sota_b4 = "compare_sota_b4.png"

def combine_matplotlib(out="COMBRIDGE_vs_SOTA.png"):
    img1 = mpimg.imread(compare_sota_b0)
    img2 = mpimg.imread(compare_sota_b4)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    # 左边图
    axes[0].imshow(img1)
    axes[0].set_title("no Byzantine nodes (b=0)", fontsize=20)
    axes[0].axis("off")

    # 右边图
    axes[1].imshow(img2)
    axes[1].set_title("with 4 Byzantine nodes (b=4)", fontsize=20)
    axes[1].axis("off")

    # 全局标签
    fig.text(0.5, 0.04, "Communication Rounds", ha="center", fontsize=20)
    fig.text(0.04, 0.5, "Mean Accuracy", va="center", rotation="vertical", fontsize=20)

    # 颜色与算法对应关系
    color_map = {
        "COMBRIDGE": "deepskyblue",
        "SecureDL": "orange",
        "RTC": "green",
        "UBAR": "red",
        "BRASO": "purple"
    }
    legend_patches = [plt.Line2D([0], [0], color=c, lw=2, label=name)
                      for name, c in color_map.items()]

    # 把 legend 放在右下角偏上一点
    axes[1].legend(
        handles=legend_patches,
        loc="lower right",
        bbox_to_anchor=(1.0, 0.25),  # 0.25 比默认 0.0 高一点
        fontsize=9,
        frameon=True,
        facecolor="white",
        framealpha=0.8
    )

    plt.subplots_adjust(wspace=0.02, left=0.08, right=0.95, top=0.9, bottom=0.12)

    fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

if __name__ == "__main__":
    combine_matplotlib()
