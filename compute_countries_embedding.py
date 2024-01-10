import pandas as pd
import matplotlib.pyplot as plt
import time
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler 

from custom_tsne import CustomTSNE

PERPLEXITY = 4.0

filepath = "datasets/country_dataset_with_names.csv"

df = pd.read_csv(filepath, index_col = 0)

data = df.to_numpy()
X = data[0:].astype(np.float64)
scaler = StandardScaler()
X = scaler.fit_transform(X)
countries = df.index.to_numpy()

time_start = time.time()

tsne = CustomTSNE(X, countries)

embedding, _, _, _ = tsne.run(no_dims = 2, perplexity = PERPLEXITY)

tsne.save_tsne_data(f"embeddings/countries/perplexity_{int(PERPLEXITY)}")

print('t-sne time: {} seconds'.format(time.time()-time_start))