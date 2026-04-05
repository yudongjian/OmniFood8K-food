
# OmniFood8K: Single-Image Nutrition Estimation via Hierarchical Frequency-Aligned Fusion


Welcome to my homepage: [OmniFood8K-food](https://yudongjian.github.io/OmniFood8K-food/).


# 🎉 This work was accepted in CVPR 2026!

---

## 🧾 Paper Information


<p align="center" style="font-size:small;">

Dongjian Yu¹, Weiqing Min², Qian Jiang¹, Xing Lin¹, Xin Jin¹, Shuqiang Jiang²

</p>

<p align="center" style="font-size:small;">

¹ Yunnan University

</p>

<p align="center" style="font-size:small;">

² Key Laboratory of Intelligent Information Processing, Institute of Computing Technology, Chinese Academy of Sciences

</p>

### Please feel free to contact me at yudongjian@stu.ynu.edu.cn if you have any questions.


## Prerequisite Step 1

Before using this project, please download the pre-trained weight files:  你首先需要下载预训练的权重文件：

[Download Swin-Transforemer, ConvNext here](https://drive.google.com/drive/folders/1i-AExbFDi4cLy_OPYUmGm_q5f8EITpjJ?usp=drive_link)

After downloading, place the files in the `pth/` folder of the project (create the folder if it doesn't exist).


## Prerequisites Step 2

Generate predicted depth maps.

- **training**:  
  In `train3D-mm.py`, please configure the following paths:
  - `clip_path`  (located at **line 81**).
  - `pth_path` (for **Swin-T** and **ConvNeXt** pre-trained weights) (located at **line 94**).
  - `checkpoint` (for **Point Transformer** ) (located at **line 131**).
  In `model/three_D.py`, please set the path to the **DINO** pre-trained weights (located at **line 174**).



## 🚧 Train
```bash
train_nutrition.py --b 8 --log ./logs/test3
```
