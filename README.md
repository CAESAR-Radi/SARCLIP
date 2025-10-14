# SARCLIP: Multimodal Foundation Model for SAR Imagery

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-brightgreen.svg)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-orange.svg)](https://developer.nvidia.com/cuda-12-1-0-download-archive)

## 🚀 Overview

**SARCLIP** is a multimodal foundation model specifically designed for Synthetic Aperture Radar (SAR) imagery based on the Contrastive Language-Image Pre-training (CLIP) framework. SARCLIP enables cross-modal understanding between SAR images and textual information, supporting zero-shot classification, cross-modal retrieval, and image-text inference.

---

## 🛠 Installation

### Environment Requirements

- **Operating System:** Linux or Windows
- **Python:** ≥ 3.8
- **CUDA:** Compatible CUDA version as supported by PyTorch

### Dependencies
Install required Python libraries:

```bash
pip install -r requirements.txt
```

### Hardware Recommendations

- **GPU:** NVIDIA RTX3060 or higher
- **Memory:** ≥ 16GB RAM
- **VRAM:** ≥ 12GB GPU Memory
- **Disk:** ≥ 30GB free disk space

---

## 📂 Project Structure
```
SARCLIP-main/
├── sar_clip/
│   ├── model_configs/     # Model configs & pre-trained weights
│   ├── *.py               # Core model code
├── data/                  # Dataset directory
├── retrieval.py           # Cross-modal retrieval script
├── zero-shot.py           # Zero-shot classification script
├── zero-shot-inference.py # Image-text inference script
├── example.py             # Demonstration script
├── requirements.txt
├── README.md
```

---

## 🚩 Quick Start

### Zero-Shot Classification

Update `CLASSNAMES` and `TEMPLATES` in `zero-shot.py`, then execute:

```bash
python zero-shot.py \
  --imagenet-val "./data/zero-shot" \
  --batch-size 8 \
  --model "ViT-B-32" \
  --cache-dir "./sar_clip/model_configs/ViT-B-32" \
  --pretrained "./sar_clip/model_configs/ViT-B-32/vit_b_32_model.safetensors"
```

### Cross-Modal Retrieval

Execute the retrieval script (Extract the ./data/retrieval/retrieval.rar file first):

```bash
python retrieval.py \
  --val-data "./data/retrieval_file_list.csv" \
  --csv-img-key "filename" \
  --csv-caption-key "caption" \
  --batch-size 8 \
  --model "ViT-B-32" \
  --cache-dir "./sar_clip/model_configs/ViT-B-32" \
  --pretrained "./sar_clip/model_configs/ViT-B-32/vit_b_32_model.safetensors"
```

### Image-Text Inference

Run inference directly on images:

```bash
python zero-shot-inference.py \
  --image-dir "path/to/images" \
  --batch-size 8 \
  --model "ViT-B-32" \
  --cache-dir "./sar_clip/model_configs/ViT-B-32" \
  --pretrained "./sar_clip/model_configs/ViT-B-32/vit_b_32_model.safetensors"
```

---

### Example Output
Running `example.py` provides a visualization and outputs textual predictions:

```
Predictions:
- an SAR image of urban zones                        1.0000
- an SAR image of water areas                        0.0000
- an SAR image of croplands                          0.0000
- one solitary marine craft is visible in the right region . 0.0000
- along the right side , several storage tanks are be detected . 0.0000
- 1 aircraft is found throughout the frame .         0.0000
```

---

## ❓ Troubleshooting

- **Out of Memory (OOM):** Decrease `--batch-size`.
- **Model Loading Failed:** Verify the correct path to the pretrained model.
- **GPU Not Used:** Ensure CUDA and PyTorch compatibility.

---

## 📌 License

- **Code**: Released under the [MIT License](./LICENSE).  
- **Dataset (SARCAP)**: Released under a separate [Dataset License](./DATASET_LICENSE.md), for non-commercial research and educational use only.

---

## 💾 Model Weights & Dataset Access

### Pretrained Model Weights

The pretrained **SARCLIP** weights are publicly available for research and non-commercial use.

- **SARCLIP Weights:** [🔗 Baidu Netdisk](https://pan.baidu.com/s/1RjS--72GHFynCqE5HctXRw?pwd=dizf) (Extraction code: `dizf`)  

To use the pretrained weights, place them under:
```bash
./sar_clip/model_configs/{MODEL_NAME}/
```

### Dataset Access

All released data are intended for non-commercial research and educational purposes only.

- **SARCAP Dataset:** [🔗 Baidu Netdisk](https://pan.baidu.com/s/1iuRCOfEtJFnvjyVVsdwnYg?pwd=2nxm) (Extraction code: `2nxm`)  
- **Zero-Shot:** [🔗 Baidu Netdisk](https://pan.baidu.com/s/1Yjzf0j0fFQH82G7FVBT41A?pwd=quh2) (Extraction code: `quh2`)

Dataset structure:
```
SARCAP/
├── img/                   # SAR image patches
├── img_caption.csv        # Image-text pairs
```
To use the zero-shot examples, place them under:
```bash
./data/zero-shot/
```

---

## 📚 Citation

If you use SARCLIP, please cite:

```bibtex
@misc{SARCLIP2025,
  author = {CAESAR-Radi},
  title = {SARCLIP: A Multimodal Foundation Framework for SAR Imagery via Contrastive Language-Image Pre-Training},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/CAESAR-Radi/SARCLIP}
}
```

---

## 🌟 Acknowledgements

We thank the following organizations for providing datasets and inspiration:
- Capella Space (Capella SAR Data)
- ESA Copernicus Programme (WorldCover)
- Anhui University (OGSOD)
- University of Electronic Science and Technology of China (RSDD)
- Huazhong University of Science and Technology (SADD)
- Chinese Academy of Sciences (SIVED)
- Technical University of Munich (SEN12MS)

Special thanks to the [OpenCLIP](https://github.com/mlfoundations/open_clip) team for their significant contributions.

