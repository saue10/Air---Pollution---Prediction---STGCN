# Air Pollution Prediction using STGCN

## 📌 Overview
This project focuses on predicting PM2.5 air pollution levels using a spatio-temporal deep learning model. The model combines Graph Convolutional Networks (GCN), LSTM, and attention mechanisms.

## 🚀 Key Features
- Spatio-temporal modeling using GCN + LSTM
- Attention-based dynamic adjacency
- Wavelet-based signal denoising (advanced)
- Mini-batch training for efficiency
- Dual evaluation (Normalized MAE & Real MAE)

## 📊 Results
- Normalized MAE: ~0.22 ✅
- Real MAE: ~5.5 µg/m³

## 🧠 Methodology
1. Data preprocessing and normalization  
2. Graph construction (spatial relationships)  
3. Temporal sequence modeling  
4. Attention-based dynamic graph learning  
5. Wavelet transform for noise reduction  
6. Model training using PyTorch  

## 🛠 Technologies Used
- Python  
- PyTorch  
- NumPy, Pandas  
- PyWavelets  

## ▶️ How to Run
```bash
python step1_data.py
python step2_graph.py
python step3_sequence.py
python train_full_model.py

⚠️ Note
Dataset is not included due to large size. Please use your own dataset.

👨‍💻 Author
Saurabh Rai
