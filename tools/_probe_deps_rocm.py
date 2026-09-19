import importlib.util
print("numpy", importlib.util.find_spec("numpy") is not None)
print("torch", importlib.util.find_spec("torch") is not None)
print("imageio", importlib.util.find_spec("imageio") is not None)
try:
    import imageio.v2 as imageio
    print("imageio_ok", imageio.__version__)
except Exception as e:
    print("imageio_err", e)
