import pandas as pd
import numpy as np
import pickle
import pywt

def wavelet_denoise(signal):
    signal = np.array(signal, dtype=float).copy()   # ✅ FIX HERE

    coeffs = pywt.wavedec(signal, 'db4', level=2)

    # thresholding
    coeffs[1:] = [pywt.threshold(c, value=0.5*np.std(c)) for c in coeffs[1:]]

    return pywt.waverec(coeffs, 'db4')[:len(signal)]   # trim to original length

df = pd.read_csv("cleaned_data.csv")

features = ["PM2.5", "PM10", "NO2", "CO", "AT", "RH", "WS", "WD", "TOT-RF"]

df = df.dropna()
df["PM2.5"]=df.groupby("Site")["PM2.5"].transform(wavelet_denoise)

# Save PM2.5 stats BEFORE normalization
pm25_mean = df["PM2.5"].mean()
pm25_std = df["PM2.5"].std()

# Normalize all features
df[features] = (df[features] - df[features].mean()) / df[features].std()

# Save scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump((pm25_mean, pm25_std), f)


# Create station_id (IMPORTANT)
df["station_id"] = df["Site"].astype("category").cat.codes

print(df.head())
df.to_csv("processed_data.csv", index=False)
print("Processed data saved!")