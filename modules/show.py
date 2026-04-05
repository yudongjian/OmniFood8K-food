import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import os


# ==========================================
# 2. 核心绘图函数 (点群 + 分布)
# ==========================================
def visualize_alignment_results(rgb_feat, depth_feat, stage_name="Stage 4", save_dir="results"):
    """
    在一个画布上展示 t-SNE 点群图和 KDE 密度分布图
    """

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 转换数据
    r = rgb_feat.detach().cpu().numpy()
    d = depth_feat.detach().cpu().numpy()

    # --- 布局设置 ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # --- 图 1: t-SNE 点群流形 (Manifold) ---
    all_feats = np.concatenate([r, d], axis=0)
    # 动态调整困惑度以防报错
    perp = min(30, max(1, all_feats.shape[0] - 1))
    tsne = TSNE(n_components=2, perplexity=perp, init='pca', random_state=42)
    feats_2d = tsne.fit_transform(all_feats)

    ax1.scatter(feats_2d[:len(r), 0], feats_2d[:len(r), 1],
                c='#FF4B4B', label='RGB Features', alpha=0.6, edgecolors='w', s=60)
    ax1.scatter(feats_2d[len(r):, 0], feats_2d[len(r):, 1],
                c='#1C90FF', label='Depth Features', alpha=0.6, edgecolors='w', s=60, marker='s')

    ax1.set_title(f"Feature Manifold (t-SNE) - {stage_name}", fontsize=14)
    ax1.legend()

    # --- 图 2: KDE 密度分布 (Density) ---
    # 取特征的第一主成分或者特征范数来代表分布
    r_norm = np.linalg.norm(r, axis=1)
    d_norm = np.linalg.norm(d, axis=1)

    sns.kdeplot(r_norm, ax=ax2, fill=True, color="#FF4B4B", label="RGB Density", alpha=0.4)
    sns.kdeplot(d_norm, ax=ax2, fill=True, color="#1C90FF", label="Depth Density", alpha=0.4)

    ax2.set_title(f"Probability Density Alignment - {stage_name}", fontsize=14)
    ax2.set_xlabel("Feature Intensity (L2 Norm)")
    ax2.set_ylabel("Probability Density")
    ax2.legend()

    plt.suptitle(f"Multi-modal Alignment Analysis: {stage_name}", fontsize=16, y=1.05)
    plt.tight_layout()
    # --- 保存逻辑 ---
    # 建议保存为 PDF（矢量图，投稿首选）和 PNG（预览用）
    file_name = stage_name.replace(" ", "_").lower()

    # # 1. 保存为 PNG 供查看
    # plt.savefig(os.path.join(save_dir, f"{file_name}_alignment.png"), dpi=300, bbox_inches='tight')
    # 2. 保存为 PDF 用于 LaTeX 投稿 (永不失真)
    plt.savefig(os.path.join(save_dir, f"{file_name}_alignment.pdf"), bbox_inches='tight')


# ==========================================
# 3. 运行演示
# ==========================================
if __name__ == "__main__":
    print("[INFO] 正在展示：未对齐时的分布情况...")
    
    # 模拟未对齐的情况
    x = torch.randn(800, 1024)
    y = torch.randn(800, 1024)
    visualize_alignment_results(x, y, "Stage 4 (After MMD)")