import pandas as pd
import matplotlib.pyplot as plt
import time
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler 

from custom_tsne import CustomTSNE

PERPLEXITY = 30.0

filepath = "datasets/column_3C.xls"

df = pd.read_csv(filepath)

X = df.loc[:, df.columns != "class"].to_numpy().astype(np.float64)
scaler = StandardScaler()
X = scaler.fit_transform(X)
targets = df["class"].to_numpy()

time_start = time.time()

tsne = CustomTSNE(X, targets)
embedding, _, _, _ = tsne.run(no_dims = 2, perplexity = PERPLEXITY)

tsne.save_tsne_data(f"embeddings/vertebral/perplexity_{int(PERPLEXITY)}")

print('t-sne time: {} seconds'.format(time.time()-time_start))