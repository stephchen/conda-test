import wandb
import os
import time
import sys
import torch


wandb.init()

print('launch test!')
print('sleeping.......')
time.sleep(5)
print('awake!')

print(sys.version_info)

import platform
print(platform.system(), platform.release(), platform.linux_distribution())

print("cuda:", torch.cuda.is_available())
count = torch.cuda.device_count()
if count > 0:
    print(torch.cuda.get_device_name(0))
print("cuda version:", torch.version.cuda)
