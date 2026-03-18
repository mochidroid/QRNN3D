import sys
print("Starting...", flush=True)

import os
print("Imported os", flush=True)

try:
    import torch
    print("Imported torch:", torch.__version__, flush=True)
except Exception as e:
    print("Torch import error:", e, flush=True)

try:
    import scipy.io as sio
    print("Imported scipy", flush=True)
except Exception as e:
    print("Scipy import error:", e, flush=True)

try:
    import models
    print("Imported models", flush=True)
except Exception as e:
    print("Models import error:", e, flush=True)

print("Done", flush=True)
