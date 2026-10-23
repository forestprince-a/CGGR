"""Lazy CPU/CUDA selection. CUDA requests never silently fall back to CPU."""
import numpy as np


def array_module(value):
    if type(value).__module__.split(".")[0] in ("cupy", "cupyx"):
        import cupy
        return cupy
    return np


def to_host(value):
    if type(value).__module__.split(".")[0] in ("cupy", "cupyx"):
        return value.get()
    return value


def select_device(device):
    if not isinstance(device, str) or not (
            device in ("cpu", "cuda") or
            (device.startswith("cuda:") and device[5:].isdigit())):
        raise ValueError("device must be cpu, cuda, or cuda:<index>")
    if device == "cpu":
        return np
    try:
        import cupy as cp
        index = int(device.split(":")[1]) if ":" in device else 0
        cp.cuda.Device(index).use()
        # Verify the runtime before loading data or beginning a long batch.
        cp.empty(1, dtype=cp.float64)
        return cp
    except Exception as exc:
        raise RuntimeError(
            "CUDA backend unavailable. Install the CuPy build matching the server's "
            "CUDA runtime (see requirements-gpu.txt), and verify the selected GPU. "
            "No CPU fallback was performed."
        ) from exc


def synchronize(xp):
    if xp is not np:
        xp.cuda.get_current_stream().synchronize()


def device_info(xp):
    if xp is np:
        return {"backend": "numpy", "device": "cpu", "dtype": "float64"}
    index = xp.cuda.Device().id
    props = xp.cuda.runtime.getDeviceProperties(index)
    name = props["name"]
    if isinstance(name, bytes):
        name = name.decode()
    return {"backend": "cupy", "device": f"cuda:{index}", "name": name,
            "memory_bytes": int(props["totalGlobalMem"]), "dtype": "float64",
            "cupy_version": xp.__version__,
            "cuda_runtime": int(xp.cuda.runtime.runtimeGetVersion()),
            "cuda_driver": int(xp.cuda.runtime.driverGetVersion())}
