import os

# On macOS the torch and lightgbm wheels each bundle a libomp, and one process holding both aborts with
# "OMP: Error #15" (in either import order). The suite imports both, so allow the duplicate runtime here, import
# lightgbm first, and keep torch to one thread so its pool does not wait on a runtime LightGBM already started.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import lightgbm  # noqa: E402,F401
import torch  # noqa: E402

torch.set_num_threads(1)
