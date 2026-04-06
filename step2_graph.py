import pandas as pd
import torch
import numpy as np

df = pd.read_csv("processed_data.csv")

features = ["PM2.5", "PM10", "NO2", "CO", "AT", "RH"]

df = df.dropna()

# station_id mapping
df["station_id"] = df["Site"].astype("category").cat.codes

num_nodes = df["station_id"].nunique()

print("Number of stations:", num_nodes)

# Sort by time + station
df = df.sort_values(["From Date", "station_id"])

# Create time steps
grouped = df.groupby("From Date")

data_list = []

for _, group in grouped:
    if len(group) == num_nodes:
        data_list.append(group[features].values)

data = np.array(data_list)

data = torch.tensor(data, dtype=torch.float32)

print("Graph data shape:", data.shape)
torch.save(data, "graph_data.pt")
print("Graph data saved!")