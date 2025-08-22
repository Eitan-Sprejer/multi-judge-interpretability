#%%
import os
import sys

ROOT = os.path.join(os.getcwd(), '..')
sys.path.append(ROOT)

#%%
from pipeline.core.dataset_loader import DatasetLoader

dataset_loader = DatasetLoader()

#%%
personas = dataset_loader.load_existing_personas(os.path.join(ROOT, 'data/data_with_all_personas.pkl'))
ultrafeedback = dataset_loader.load_ultrafeedback(n_samples=1000, random_seed=42)

# %%
