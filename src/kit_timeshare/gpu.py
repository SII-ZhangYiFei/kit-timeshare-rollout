"""Discover physical devices and fail closed before importing CUDA/Kit."""

from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
import csv, io, os, re, stat, subprocess, sys


@dataclass(frozen=True)
class GPU:
    uuid: str
    pci_bus: str
    minor: int
    render_node: str
    name: str
    driver: str


def discover():
    try:
        text = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=uuid,pci.bus_id,name,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "NVIDIA driver/nvidia-smi unavailable; native backends require Linux and NVIDIA RTX"
        ) from exc
    result = []
    for row in csv.reader(io.StringIO(text)):
        uuid, bus, name, driver = (v.strip() for v in row)
        parts = bus.lower().split(":")
        bus = f"{int(parts[0],16):04x}:{parts[1]}:{parts[2]}"
        info = Path("/proc/driver/nvidia/gpus") / bus / "information"
        match = re.search(r"Device Minor:\s*(\d+)", info.read_text())
        if not match:
            raise RuntimeError(f"Cannot resolve device minor for {bus}")
        nodes = [
            str(p)
            for p in Path("/dev/dri").glob("renderD*")
            if (Path("/sys/class/drm") / p.name / "device").resolve().name == bus
        ]
        result.append(
            GPU(
                uuid,
                bus,
                int(match.group(1)),
                nodes[0] if len(nodes) == 1 else "",
                name,
                driver,
            )
        )
    return result


def accessible_nodes():
    # nvidia-smi/nvidia-modprobe may create nodes for cgroup-denied GPUs.
    # Presence alone is not visibility. Opening a node does not create a CUDA
    # context; device cgroup/permission denial must still exclude it.
    result = []
    for p in Path("/dev").glob("nvidia[0-9]*"):
        if not re.fullmatch(r"nvidia\d+", p.name) or not p.is_char_device():
            continue
        try:
            fd = os.open(p, os.O_RDWR | os.O_CLOEXEC)
        except PermissionError:
            continue
        else:
            os.close(fd)
            result.append(str(p))
    return sorted(result)


def select_isolated_gpu(uuid=None):
    devices = discover()
    nodes = accessible_nodes()
    candidates = [
        g
        for g in devices
        if f"/dev/nvidia{g.minor}" in nodes and (uuid is None or uuid == g.uuid)
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            "Expose exactly one target NVIDIA GPU, or specify gpu_uuid; use doctor/docker-command for multi-GPU hosts"
        )
    gpu = candidates[0]
    render = sorted(str(p) for p in Path("/dev/dri").glob("renderD*"))
    if (
        nodes != [f"/dev/nvidia{gpu.minor}"]
        or not gpu.render_node
        or render != [gpu.render_node]
    ):
        raise RuntimeError(
            "Multiple GPU/render device nodes are visible. CUDA_VISIBLE_DEVICES alone does not isolate Vulkan. Run in a GPU-specific container (see docker-command)."
        )
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized():
        if os.environ.get("CUDA_VISIBLE_DEVICES") != gpu.uuid:
            raise RuntimeError(
                "Select/isolate the GPU before initializing PyTorch CUDA"
            )
    os.environ.update(
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        CUDA_VISIBLE_DEVICES=gpu.uuid,
        OMNIGIBSON_GPU_ID="0",
        NVIDIA_VISIBLE_DEVICES=gpu.uuid,
    )
    return gpu


def docker_command(uuid, image, command, *, package_dir=None):
    gpu = next((g for g in discover() if g.uuid == uuid), None)
    if gpu is None or not gpu.render_node:
        raise ValueError("GPU UUID or DRM render device not found")
    args = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "device=" + gpu.uuid,
        "--device",
        gpu.render_node,
        "-e",
        "NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics",
        "-e",
        "CUDA_VISIBLE_DEVICES=" + gpu.uuid,
        "-e",
        "OMNIGIBSON_GPU_ID=0",
        "--shm-size=2g",
    ]
    if package_dir:
        args += [
            "--mount",
            f"type=bind,src={Path(package_dir).resolve()},dst=/opt/kit-timeshare,readonly",
        ]
    return args + [image] + list(command)
