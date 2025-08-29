# 🌌 SARCLIP

**SARCLIP: A Multimodal Foundation Model for SAR Imagery via Contrastive Language-Image Pre-Training**

## 🔥 News
- **Aug 2025**: The revised manuscript of SARCLIP has been submitted after major revision.  
- **Planned**: Code and pre-trained models will be released once the review process moves forward smoothly.  
- **Upon Acceptance**: The SARCLIP dataset will be released to ensure compliance with publication policy.  

## 🛰️ Overview

SARCLIP is the first large-scale multimodal pre-trained model tailored for Synthetic Aperture Radar (SAR) imagery. By aligning SAR images with natural language descriptions, SARCLIP achieves remarkable semantic understanding capabilities and demonstrates state-of-the-art performance across a variety of downstream tasks, including:

- 🔍 Cross-modal Retrieval  
- 🧠 Zero-shot & Few-shot Classification  
- 🔢 Object Counting  
- 📍 Spatial Localization  

This project also introduces SARTEX, an annotation framework that automatically converts structured SAR data into natural language descriptions, resulting in a dataset of approximately 400,000 image-text pairs.

## 🧠 Key Contributions

- ✅ Construction of the first large-scale SAR image-text dataset (400K pairs; resolutions from 0.1m to 20m; covering multi-source SAR data)  
- ✅ Design of a SAR-specific multimodal pre-training model, SARCLIP (compatible with ResNet and ViT architectures)  
- ✅ Proposal of the SARTEX image captioning method to enhance semantic representation  
- ✅ Extensive validation on public benchmarks, demonstrating strong performance in zero-shot, few-shot, counting, and localization tasks  

## 📦 Coming Soon

Upon acceptance of the paper, we will release the following:

- 📁 Pre-trained SARCLIP model weights (including ViT-L-14 and ResNet variants)  
- 📊 A cleaned, standardized, and diversified SAR image-text dataset  
- 🔧 Running scripts, with configurations for model using and tasks  
