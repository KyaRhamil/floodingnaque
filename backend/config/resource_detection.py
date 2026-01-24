"""
Floodingnaque Resource Detection
================================

Auto-detects system resources for optimal configuration of training parameters.

Features:
- CPU core detection (physical and logical)
- Memory detection and safe limits
- GPU detection (CUDA, ROCm)
- Intelligent default calculation
- Platform-specific optimizations

Usage:
    from config.resource_detection import (
        detect_resources,
        get_optimal_workers,
        get_safe_memory_limit,
        ResourceInfo
    )

    # Get full resource info
    resources = detect_resources()
    print(f"CPUs: {resources.cpu_count}, Memory: {resources.memory_gb}GB")

    # Get optimal worker count
    workers = get_optimal_workers()  # e.g., cpu_count - 1

    # Get safe memory limit
    memory_gb = get_safe_memory_limit()  # e.g., 80% of available

Configuration Integration:
    When max_workers: -1 or max_memory_gb: -1 in config, these values
    are automatically replaced with detected optimal values.
"""

import logging
import os
import platform
import subprocess  # nosec B404 - required for system resource detection
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class GPUInfo:
    """Information about a GPU device."""

    index: int
    name: str
    memory_mb: int
    compute_capability: Optional[str] = None
    driver_version: Optional[str] = None
    cuda_version: Optional[str] = None

    @property
    def memory_gb(self) -> float:
        return self.memory_mb / 1024


@dataclass
class ResourceInfo:
    """System resource information."""

    # CPU information
    cpu_count_physical: int
    cpu_count_logical: int
    cpu_freq_mhz: Optional[float] = None
    cpu_model: Optional[str] = None

    # Memory information
    memory_total_bytes: int = 0
    memory_available_bytes: int = 0
    swap_total_bytes: int = 0

    # GPU information
    gpus: List[GPUInfo] = field(default_factory=list)
    cuda_available: bool = False

    # Platform information
    os_name: str = ""
    os_version: str = ""
    python_version: str = ""

    @property
    def cpu_count(self) -> int:
        """Get logical CPU count (default for most use cases)."""
        return self.cpu_count_logical

    @property
    def memory_total_gb(self) -> float:
        """Get total memory in GB."""
        return self.memory_total_bytes / (1024**3)

    @property
    def memory_available_gb(self) -> float:
        """Get available memory in GB."""
        return self.memory_available_bytes / (1024**3)

    @property
    def has_gpu(self) -> bool:
        """Check if GPU is available."""
        return len(self.gpus) > 0 and self.cuda_available

    @property
    def total_gpu_memory_gb(self) -> float:
        """Get total GPU memory across all GPUs."""
        return sum(gpu.memory_gb for gpu in self.gpus)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cpu": {
                "physical_cores": self.cpu_count_physical,
                "logical_cores": self.cpu_count_logical,
                "frequency_mhz": self.cpu_freq_mhz,
                "model": self.cpu_model,
            },
            "memory": {
                "total_gb": round(self.memory_total_gb, 2),
                "available_gb": round(self.memory_available_gb, 2),
                "swap_gb": round(self.swap_total_bytes / (1024**3), 2),
            },
            "gpu": {
                "available": self.has_gpu,
                "count": len(self.gpus),
                "cuda_available": self.cuda_available,
                "total_memory_gb": round(self.total_gpu_memory_gb, 2),
                "devices": [
                    {
                        "index": gpu.index,
                        "name": gpu.name,
                        "memory_gb": round(gpu.memory_gb, 2),
                    }
                    for gpu in self.gpus
                ],
            },
            "platform": {
                "os": self.os_name,
                "version": self.os_version,
                "python": self.python_version,
            },
        }


def detect_cpu_info() -> Tuple[int, int, Optional[float], Optional[str]]:
    """
    Detect CPU information.

    Returns:
        Tuple of (physical_cores, logical_cores, frequency_mhz, model_name)
    """
    import multiprocessing

    logical_cores = multiprocessing.cpu_count()
    physical_cores = logical_cores  # Default assumption
    freq_mhz = None
    model_name = None

    # Try to get more detailed info
    try:
        import psutil

        # Physical cores
        physical_cores = psutil.cpu_count(logical=False) or logical_cores

        # CPU frequency
        freq = psutil.cpu_freq()
        if freq:
            freq_mhz = freq.current

    except ImportError:
        logger.debug("psutil not available for detailed CPU info")

    # Try to get CPU model name
    system = platform.system()

    if system == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        model_name = line.split(":")[1].strip()
                        break
        except Exception:  # nosec B110 - fallback to defaults on failure
            pass

    elif system == "Windows":
        try:
            import winreg

            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            model_name = winreg.QueryValueEx(key, "ProcessorNameString")[0]
            winreg.CloseKey(key)
        except Exception:  # nosec B110 - fallback to defaults on failure
            pass

    elif system == "Darwin":  # macOS
        try:
            result = subprocess.run(  # nosec B603 B607 - hardcoded safe command
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
            )
            if result.returncode == 0:
                model_name = result.stdout.strip()
        except Exception:  # nosec B110 - fallback to defaults on failure
            pass

    return physical_cores, logical_cores, freq_mhz, model_name


def detect_memory_info() -> Tuple[int, int, int]:
    """
    Detect memory information.

    Returns:
        Tuple of (total_bytes, available_bytes, swap_bytes)
    """
    total = 0
    available = 0
    swap = 0

    try:
        import psutil

        mem = psutil.virtual_memory()
        total = mem.total
        available = mem.available

        swap_info = psutil.swap_memory()
        swap = swap_info.total

    except ImportError:
        logger.debug("psutil not available for memory detection")

        # Fallback methods
        system = platform.system()

        if system == "Linux":
            try:
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            total = int(line.split()[1]) * 1024  # KB to bytes
                        elif line.startswith("MemAvailable:"):
                            available = int(line.split()[1]) * 1024
                        elif line.startswith("SwapTotal:"):
                            swap = int(line.split()[1]) * 1024
            except Exception:  # nosec B110 - fallback to defaults on failure
                pass

        elif system == "Windows":
            try:
                import ctypes

                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]

                stat = MEMORYSTATUSEX()
                stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))

                total = stat.ullTotalPhys
                available = stat.ullAvailPhys
                swap = stat.ullTotalPageFile - stat.ullTotalPhys

            except Exception:  # nosec B110 - fallback to defaults on failure
                pass

    return total, available, swap


def detect_gpu_info() -> Tuple[List[GPUInfo], bool]:
    """
    Detect GPU information.

    Returns:
        Tuple of (list of GPUInfo, cuda_available)
    """
    gpus = []
    cuda_available = False

    # Try PyTorch first
    try:
        import torch  # type: ignore[import-not-found]

        cuda_available = torch.cuda.is_available()

        if cuda_available:
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpus.append(
                    GPUInfo(
                        index=i,
                        name=props.name,
                        memory_mb=props.total_memory // (1024 * 1024),
                        compute_capability=f"{props.major}.{props.minor}",
                    )
                )

            # Get CUDA version
            if gpus:
                gpus[0].cuda_version = torch.version.cuda

        return gpus, cuda_available

    except ImportError:
        pass

    # Try nvidia-smi
    try:
        result = subprocess.run(  # nosec B603 B607 - hardcoded safe command for GPU detection
            ["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode == 0:
            cuda_available = True
            for line in result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    gpus.append(
                        GPUInfo(
                            index=int(parts[0]),
                            name=parts[1],
                            memory_mb=int(float(parts[2])),
                            driver_version=parts[3],
                        )
                    )

    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return gpus, cuda_available


def detect_resources() -> ResourceInfo:
    """
    Detect all system resources.

    Returns:
        ResourceInfo with detected values
    """
    # CPU
    physical_cores, logical_cores, freq_mhz, cpu_model = detect_cpu_info()

    # Memory
    mem_total, mem_available, swap_total = detect_memory_info()

    # GPU
    gpus, cuda_available = detect_gpu_info()

    # Platform
    import sys

    return ResourceInfo(
        cpu_count_physical=physical_cores,
        cpu_count_logical=logical_cores,
        cpu_freq_mhz=freq_mhz,
        cpu_model=cpu_model,
        memory_total_bytes=mem_total,
        memory_available_bytes=mem_available,
        swap_total_bytes=swap_total,
        gpus=gpus,
        cuda_available=cuda_available,
        os_name=platform.system(),
        os_version=platform.release(),
        python_version=sys.version.split()[0],
    )


def get_optimal_workers(
    task_type: str = "cpu_bound", leave_free: int = 1, max_workers: Optional[int] = None, use_physical: bool = False
) -> int:
    """
    Get optimal number of worker processes.

    Args:
        task_type: Type of task ("cpu_bound", "io_bound", "mixed")
        leave_free: Number of cores to leave free for system
        max_workers: Maximum workers to return (None for no limit)
        use_physical: Use physical instead of logical cores

    Returns:
        Optimal number of workers
    """
    resources = detect_resources()

    if use_physical:
        cores = resources.cpu_count_physical
    else:
        cores = resources.cpu_count_logical

    # Calculate based on task type
    if task_type == "cpu_bound":
        # CPU-bound: use most cores but leave some free
        workers = max(1, cores - leave_free)
    elif task_type == "io_bound":
        # IO-bound: can use more workers than cores
        workers = cores * 2
    else:  # mixed
        workers = cores

    # Apply maximum limit
    if max_workers is not None:
        workers = min(workers, max_workers)

    return workers


def get_safe_memory_limit(fraction: float = 0.8, min_free_gb: float = 2.0, max_limit_gb: Optional[float] = None) -> int:
    """
    Get safe memory limit for training.

    Args:
        fraction: Fraction of available memory to use (default: 80%)
        min_free_gb: Minimum GB to leave free for system
        max_limit_gb: Maximum limit in GB (None for no limit)

    Returns:
        Safe memory limit in GB (integer)
    """
    resources = detect_resources()

    # Calculate based on available memory
    available_gb = resources.memory_available_gb
    total_gb = resources.memory_total_gb

    # Use available memory if it's reasonable, otherwise use total
    if available_gb < total_gb * 0.5:
        # Less than half available, something might be wrong, use total
        base_gb = total_gb
    else:
        base_gb = available_gb

    # Apply fraction
    limit_gb = base_gb * fraction

    # Ensure minimum free memory
    limit_gb = min(limit_gb, total_gb - min_free_gb)

    # Apply maximum
    if max_limit_gb is not None:
        limit_gb = min(limit_gb, max_limit_gb)

    # Return as integer (floor)
    return max(1, int(limit_gb))


def apply_resource_detection(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply resource detection to configuration.

    Replaces -1 values in resources section with detected values.

    Args:
        config: Configuration dictionary

    Returns:
        Configuration with detected resource values
    """
    import copy

    result = copy.deepcopy(config)
    resources = result.get("resources", {})

    # Detect max_workers
    if resources.get("max_workers") == -1:
        detected_workers = get_optimal_workers(task_type="cpu_bound", leave_free=1)
        resources["max_workers"] = detected_workers
        resources["_max_workers_detected"] = True
        logger.info(f"Auto-detected max_workers: {detected_workers}")

    # Detect max_memory_gb
    if resources.get("max_memory_gb") == -1:
        detected_memory = get_safe_memory_limit(fraction=0.8, min_free_gb=2.0)
        resources["max_memory_gb"] = detected_memory
        resources["_max_memory_gb_detected"] = True
        logger.info(f"Auto-detected max_memory_gb: {detected_memory}")

    result["resources"] = resources
    return result


def get_resource_recommendations(
    model_type: str = "RandomForestClassifier", dataset_size_mb: Optional[float] = None
) -> Dict[str, Any]:
    """
    Get resource recommendations for a specific model/dataset.

    Args:
        model_type: Type of model being trained
        dataset_size_mb: Size of dataset in MB (optional)

    Returns:
        Dictionary with recommended settings
    """
    resources = detect_resources()

    recommendations = {
        "max_workers": get_optimal_workers(),
        "max_memory_gb": get_safe_memory_limit(),
        "use_gpu": resources.has_gpu,
        "notes": [],
    }

    # Model-specific recommendations
    if model_type in ("RandomForestClassifier", "RandomForestRegressor"):
        # Random Forest is embarrassingly parallel
        recommendations["notes"].append("RandomForest scales well with multiple workers")
    elif model_type in ("XGBoost", "LightGBM"):
        # Gradient boosting benefits from GPU
        if resources.has_gpu:
            recommendations["use_gpu"] = True
            recommendations["notes"].append("GPU acceleration recommended for gradient boosting")

    # Dataset size recommendations
    if dataset_size_mb:
        if dataset_size_mb > resources.memory_available_gb * 1024 * 0.5:
            recommendations["notes"].append(
                "Dataset is large relative to available memory. "
                "Consider using out-of-core training or data sampling."
            )

    # Memory recommendations
    if resources.memory_available_gb < 4:
        recommendations["notes"].append("Low available memory. Consider closing other applications.")

    return recommendations


def print_system_info() -> None:
    """Print system resource information."""
    resources = detect_resources()

    print("\n" + "=" * 60)
    print("SYSTEM RESOURCES")
    print("=" * 60)

    print(f"\nCPU:")
    print(f"  Physical cores: {resources.cpu_count_physical}")
    print(f"  Logical cores:  {resources.cpu_count_logical}")
    if resources.cpu_model:
        print(f"  Model: {resources.cpu_model}")
    if resources.cpu_freq_mhz:
        print(f"  Frequency: {resources.cpu_freq_mhz:.0f} MHz")

    print(f"\nMemory:")
    print(f"  Total:     {resources.memory_total_gb:.1f} GB")
    print(f"  Available: {resources.memory_available_gb:.1f} GB")
    if resources.swap_total_bytes:
        print(f"  Swap:      {resources.swap_total_bytes / (1024**3):.1f} GB")

    print(f"\nGPU:")
    if resources.has_gpu:
        print(f"  CUDA: Available")
        for gpu in resources.gpus:
            print(f"  [{gpu.index}] {gpu.name} - {gpu.memory_gb:.1f} GB")
    else:
        print(f"  CUDA: Not available")

    print(f"\nPlatform:")
    print(f"  OS: {resources.os_name} {resources.os_version}")
    print(f"  Python: {resources.python_version}")

    print(f"\nRecommended settings:")
    print(f"  max_workers: {get_optimal_workers()}")
    print(f"  max_memory_gb: {get_safe_memory_limit()}")

    print("=" * 60 + "\n")


# Cache for resource detection (doesn't change during runtime)
_cached_resources: Optional[ResourceInfo] = None


def get_cached_resources() -> ResourceInfo:
    """Get cached resource info (detects once per session)."""
    global _cached_resources
    if _cached_resources is None:
        _cached_resources = detect_resources()
    return _cached_resources


if __name__ == "__main__":
    print_system_info()
