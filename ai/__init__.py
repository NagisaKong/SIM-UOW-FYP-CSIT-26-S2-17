"""FYP-26-S2-17 AI pipeline."""

# Register NVIDIA pip-installed CUDA/cuDNN DLL directories so onnxruntime-gpu
# can load CUDAExecutionProvider without a system-wide CUDA Toolkit install.
import os as _os
import sys as _sys

if _sys.platform == "win32":
    try:
        import importlib.util as _ilu
        _extra_dirs = []
        for _pkg in (
            "nvidia.cudnn",
            "nvidia.cuda_runtime",
            "nvidia.cuda_nvrtc",
            "nvidia.cublas",
            "nvidia.cufft",
            "nvidia.curand",
            "nvidia.cusolver",
            "nvidia.cusparse",
            "nvidia.nvjitlink",
        ):
            _spec = _ilu.find_spec(_pkg)
            if _spec and _spec.submodule_search_locations:
                _bin = _os.path.join(_spec.submodule_search_locations[0], "bin")
                if _os.path.isdir(_bin):
                    _os.add_dll_directory(_bin)
                    _extra_dirs.append(_bin)
        if _extra_dirs:
            _os.environ["PATH"] = _os.pathsep.join(_extra_dirs) + _os.pathsep + _os.environ.get("PATH", "")
    except Exception:
        pass

from .config import AIConfig
from .pipeline import AttendancePipeline
from .store import SupabaseEmbeddingStore, EmbeddingStore

__all__ = [
    "AIConfig",
    "AttendancePipeline",
    "SupabaseEmbeddingStore",
    "EmbeddingStore",
]
