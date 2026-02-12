import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# moderate
moderate_j1 = "moderate_non_iid_j1_different_b.png"
moderate_j2 = "moderate_non_iid_j2_different_b.png"
moderate_j5 = "moderate_non_iid_j5_different_b.png"
moderate_j10 = "moderate_non_iid_j10_different_b.png"

# extreme
extreme_j1 = "non_iid_j1_different_b.png"
extreme_j2 = "non_iid_j2_different_b.png"
extreme_j5 = "non_iid_j5_different_b.png"
extreme_j10 = "non_iid_j10_different_b.png"

def combine_matplotlib(out="COMBRIDGE_vs_Non_IID.png"):
    imgs = [
        mpimg.imread(moderate_j1),
        mpimg.imread(moderate_j2),
        mpimg.imread(moderate_j5),
        mpimg.imread(moderate_j10),
        mpimg.imread(extreme_j1),
        mpimg.imread(extreme_j2),
        mpimg.imread(extreme_j5),
        mpimg.imread(extreme_j10),
    ]

    subtitles = [
        "Moderate Non-IID, j=1", "Moderate Non-IID, j=2",
        "Moderate Non-IID, j=5", "Moderate Non-IID, j=10",
        "Extreme Non-IID, j=1", "Extreme Non-IID, j=2",
        "Extreme Non-IID, j=5", "Extreme Non-IID, j=10"
    ]

    # 缩小整体尺寸
    fig, axes = plt.subplots(2, 4, figsize=(14, 6), dpi=300)
    axes = axes.flatten()

    for ax, img, subtitle in zip(axes, imgs, subtitles):
        ax.imshow(img)
        ax.set_title(subtitle, fontsize=15)
        ax.axis("off")

    # 全局标签
    fig.text(0.5, 0.03, "Communications Rounds", ha="center", fontsize=20)
    fig.text(0.04, 0.5, "Mean Accuracy", va="center", rotation="vertical", fontsize=20)

    # legend 放右边竖排
    color_map = {"b=0": "blue", "b=1": "gold", "b=2": "green", "b=4": "red"}
    legend_patches = [plt.Line2D([0], [0], color=c, lw=2, label=l) for l, c in color_map.items()]
    fig.legend(handles=legend_patches,
               loc="center right",
               bbox_to_anchor=(0.98, 0.5),
               fontsize=12)

    # 调整子图间距，减少空白
    plt.subplots_adjust(wspace=0.05, hspace=0.05, left=0.08, right=0.93, top=0.9, bottom=0.1)

    fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

if __name__ == "__main__":
    combine_matplotlib()
