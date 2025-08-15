#%%
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

# %%
from pipeline.core.judge_creation import create_or_update_judge
# %%