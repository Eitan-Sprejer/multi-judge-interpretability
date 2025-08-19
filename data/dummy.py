#%%
import pickle

with open('data_with_all_personas.pkl', 'rb') as f:
    data = pickle.load(f)

# %%
data['human_feedback'].iloc[0]
# %%
