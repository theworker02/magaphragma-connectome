import torch
print("version", torch.__version__)
print("cuda", torch.cuda.is_available())
print("hip", getattr(torch.version, "hip", None))
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
