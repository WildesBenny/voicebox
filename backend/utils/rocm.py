"""AMD ROCm / HIP GPU detection and environment configuration.

PyTorch's ROCm build exposes AMD GPUs through the ``torch.cuda`` API — HIP
masquerades as CUDA, so ``torch.cuda.is_available()`` returns ``True`` and
``device="cuda"`` runs on the Radeon GPU. The rest of the backend therefore
needs no special-casing to *use* an AMD GPU.

The one thing that does need handling is the GPU **architecture** (``gfxNNNN``).
When the installed ROCm build doesn't ship compiled kernels for the exact
architecture, HIP either refuses to initialize or silently falls back to the
CPU. The fix is to set ``HSA_OVERRIDE_GFX_VERSION`` to the nearest supported
architecture *before* the HIP runtime initializes.

This module detects the real architecture from the kernel (Linux KFD sysfs,
which needs no PyTorch and no HIP init) and only sets an override when the
running ROCm build doesn't natively support it. That makes brand-new hardware
work — most importantly the **AMD Ryzen AI Max+ 395** (Strix Halo, ``gfx1151``)
— across ROCm 6.x and 7.x, without pessimizing GPUs that are natively
supported.

Key invariant: ``configure_rocm_environment()`` must run before anything calls
into ``torch.cuda`` (which triggers HIP init and locks in the override).
"""

import functools
import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# Linux KFD topology — the kernel's authoritative view of every compute node.
# Readable without root and without initializing HIP/ROCr.
_KFD_NODES_DIR = Path("/sys/class/kfd/kfd/topology/nodes")

# AMD's PCI vendor id (0x1002) as reported by KFD's ``vendor_id`` property.
_AMD_VENDOR_ID = 4098

# Minimum ROCm (major, minor) whose PyTorch wheels ship compiled kernels for a
# given architecture. An architecture absent from this map is community-only
# (e.g. consumer iGPUs) and always needs an HSA override.
#
# Sources: AMD ROCm "Supported GPUs" matrix and the architectures present in
# the official ``download.pytorch.org/whl/rocmX.Y`` wheels.
_NATIVE_MIN_ROCM: dict[str, tuple[int, int]] = {
    # CDNA / Instinct
    "gfx908": (5, 0),   # MI100
    "gfx90a": (5, 2),   # MI200
    "gfx942": (6, 0),   # MI300
    "gfx950": (6, 4),   # MI350 (CDNA4)
    # RDNA2
    "gfx1030": (5, 5),  # RX 6800/6900, PRO W6800
    # RDNA3
    "gfx1100": (5, 6),  # RX 7900 XTX/XT, PRO W7900
    "gfx1101": (6, 1),  # RX 7800/7700 XT, PRO W7700
    "gfx1102": (6, 1),  # RX 7600
    # RDNA3.5 (Strix / Strix Halo APUs)
    "gfx1150": (6, 4),  # Strix Point — Ryzen AI 9 HX 3xx
    "gfx1151": (6, 4),  # Strix Halo — Ryzen AI Max+ 395
    # RDNA4
    "gfx1200": (6, 3),  # RX 9060
    "gfx1201": (6, 3),  # RX 9070 (XT), AI PRO R9700
}

# HSA_OVERRIDE_GFX_VERSION target (dot notation) per architecture *major*, used
# when the real architecture isn't natively supported. The override maps an
# unsupported chip onto the closest base architecture of the same ISA family
# that every ROCm build of that generation ships:
#   gfx9xx  -> 9.0.0  (Vega / GCN5)
#   gfx10xx -> 10.3.0 (gfx1030, RDNA2)
#   gfx11xx -> 11.0.0 (gfx1100, RDNA3 — also correct for RDNA3.5 Strix Halo)
#   gfx12xx -> 12.0.0 (gfx1200, RDNA4)
_FAMILY_OVERRIDE: dict[int, str] = {
    9: "9.0.0",
    10: "10.3.0",
    11: "11.0.0",
    12: "12.0.0",
}

_GFX_RE = re.compile(r"^gfx([0-9a-f]+)$")


def _decode_gfx_target_version(version: int) -> str | None:
    """Decode a KFD ``gfx_target_version`` integer into a ``gfxNNNN`` string.

    ROCm encodes the architecture as ``major * 10000 + minor * 100 + step``
    where ``minor`` and ``step`` are rendered as single hex digits. For
    example ``110501`` -> ``gfx1151`` (Strix Halo) and ``90010`` -> ``gfx90a``.

    Args:
        version: The raw ``gfx_target_version`` value from KFD sysfs.

    Returns:
        The ``gfxNNNN`` architecture string, or ``None`` for non-GPU nodes
        (CPU topology nodes report ``0``).
    """
    if version <= 0:
        return None
    major = version // 10000
    minor = (version // 100) % 100
    step = version % 100
    return f"gfx{major}{minor:x}{step:x}"


def _read_node_properties(node_dir: Path) -> dict[str, int]:
    """Parse a KFD node ``properties`` file into an int-valued mapping.

    Args:
        node_dir: A ``/sys/class/kfd/kfd/topology/nodes/<N>`` directory.

    Returns:
        Mapping of property name to integer value. Empty on any read error.
    """
    props: dict[str, int] = {}
    try:
        text = (node_dir / "properties").read_text()
    except OSError:
        return props
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            try:
                props[parts[0]] = int(parts[1], 0)
            except ValueError:
                continue
    return props


@functools.lru_cache(maxsize=1)
def detect_amd_gpu_archs() -> tuple[str, ...]:
    """Detect the ``gfx`` architecture of every AMD GPU via Linux KFD sysfs.

    This reads the kernel topology directly, so it needs neither PyTorch nor a
    HIP runtime initialization and is safe to call before ``HSA_OVERRIDE_GFX_VERSION``
    is set. On non-Linux platforms (no KFD) it returns an empty tuple.

    Returns:
        Architectures in node order, e.g. ``("gfx1151",)`` for a Ryzen AI
        Max+ 395. Empty when no AMD GPU is present.
    """
    if not _KFD_NODES_DIR.is_dir():
        return ()

    archs: list[str] = []
    try:
        node_dirs = sorted(_KFD_NODES_DIR.iterdir(), key=lambda p: p.name)
    except OSError:
        return ()

    for node_dir in node_dirs:
        props = _read_node_properties(node_dir)
        if props.get("vendor_id") != _AMD_VENDOR_ID:
            continue
        # GPU nodes have SIMDs; CPU nodes report simd_count 0 / gfx_target_version 0.
        if props.get("simd_count", 0) <= 0:
            continue
        arch = _decode_gfx_target_version(props.get("gfx_target_version", 0))
        if arch:
            archs.append(arch)

    return tuple(archs)


def _arch_major(arch: str) -> int | None:
    """Return the architecture-family major number (9, 10, 11, 12) for a gfx name."""
    match = _GFX_RE.match(arch)
    if not match:
        return None
    digits = match.group(1)
    # The trailing two hex digits are minor+step; everything before is the major.
    if len(digits) < 3:
        return None
    try:
        return int(digits[:-2])
    except ValueError:
        return None


def _is_natively_supported(arch: str, rocm_version: tuple[int, int]) -> bool:
    """Whether the installed ROCm build ships kernels for ``arch`` natively.

    Args:
        arch: Architecture string such as ``"gfx1151"``.
        rocm_version: Installed ROCm ``(major, minor)``.

    Returns:
        ``True`` if no HSA override is needed for this architecture.
    """
    minimum = _NATIVE_MIN_ROCM.get(arch)
    if minimum is None:
        return False
    return rocm_version >= minimum


def _override_for_arch(arch: str) -> str | None:
    """Return the ``HSA_OVERRIDE_GFX_VERSION`` value to use for ``arch``, if any."""
    major = _arch_major(arch)
    if major is None:
        return None
    return _FAMILY_OVERRIDE.get(major)


def is_rocm_torch() -> bool:
    """Whether the installed PyTorch is a ROCm/HIP build.

    Reading ``torch.version.hip`` does not initialize the HIP runtime, so this
    is safe to call before configuring ``HSA_OVERRIDE_GFX_VERSION``.
    """
    try:
        import torch  # lazy: heavy import

        return bool(getattr(torch.version, "hip", None))
    except Exception:
        return False


def get_rocm_version() -> tuple[int, int] | None:
    """Parse the ROCm ``(major, minor)`` the PyTorch wheel was built against.

    Returns:
        ``(6, 4)`` for a ROCm 6.4 build, or ``None`` if PyTorch is not a ROCm
        build (CUDA / CPU) or the version string can't be parsed.
    """
    try:
        import torch  # lazy: heavy import

        hip = getattr(torch.version, "hip", None)
    except Exception:
        return None
    if not hip:
        return None
    match = re.match(r"(\d+)\.(\d+)", str(hip))
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def configure_rocm_environment() -> None:
    """Configure AMD ROCm environment variables before PyTorch uses the GPU.

    Detects the AMD GPU architecture from the kernel and, when the installed
    ROCm build doesn't natively support it, sets ``HSA_OVERRIDE_GFX_VERSION``
    to the closest supported architecture so inference runs on the GPU instead
    of crashing or falling back to CPU.

    Behavior:
        * A user-provided ``HSA_OVERRIDE_GFX_VERSION`` is always respected.
        * No AMD GPU (e.g. NVIDIA, Apple Silicon, CPU-only) → does nothing.
        * Natively supported architecture (e.g. ``gfx1151`` on ROCm 6.4+) →
          does nothing, so the GPU runs its own optimized kernels.
        * Unsupported architecture → sets the family-base override.

    Must run before any ``torch.cuda`` call; once HIP initializes, the override
    is locked in for the process.
    """
    # Keep MIOpen quiet unless the user wants its logs. Harmless on non-AMD.
    os.environ.setdefault("MIOPEN_LOG_LEVEL", "4")

    if os.environ.get("HSA_OVERRIDE_GFX_VERSION"):
        logger.info(
            "ROCm: respecting user-set HSA_OVERRIDE_GFX_VERSION=%s",
            os.environ["HSA_OVERRIDE_GFX_VERSION"],
        )
        return

    archs = detect_amd_gpu_archs()
    if not archs:
        return  # No AMD GPU detected — nothing to configure.

    arch = archs[0]
    if len(set(archs)) > 1:
        logger.info("ROCm: multiple AMD GPU architectures detected %s; configuring for %s", archs, arch)

    rocm_version = get_rocm_version()
    if rocm_version is None:
        logger.warning(
            "ROCm: detected AMD GPU (%s) but PyTorch is not a ROCm build. "
            "Install ROCm PyTorch (see backend/requirements-rocm.txt) to use the GPU.",
            arch,
        )
        return

    version_str = f"{rocm_version[0]}.{rocm_version[1]}"
    if _is_natively_supported(arch, rocm_version):
        logger.info("ROCm %s: %s is natively supported; no HSA override needed.", version_str, arch)
        return

    override = _override_for_arch(arch)
    if override:
        os.environ["HSA_OVERRIDE_GFX_VERSION"] = override
        logger.info(
            "ROCm %s: %s is not natively supported; set HSA_OVERRIDE_GFX_VERSION=%s. "
            "Override with the HSA_OVERRIDE_GFX_VERSION env var if needed.",
            version_str,
            arch,
            override,
        )
    else:
        logger.warning(
            "ROCm %s: %s is not natively supported and no override is known. "
            "Set HSA_OVERRIDE_GFX_VERSION manually if the GPU fails to initialize.",
            version_str,
            arch,
        )


def check_rocm_compatibility() -> tuple[bool, str | None]:
    """Check whether the running PyTorch ROCm build can drive the AMD GPU.

    Mirrors ``backends.base.check_cuda_compatibility`` for AMD: compares the
    live device architecture against the architectures this wheel was compiled
    for. Safe to call after the GPU is in use.

    Returns:
        ``(compatible, warning_message)``. ``compatible`` is ``True`` when the
        GPU is usable (or there is no ROCm GPU); ``warning_message`` is a
        human-readable hint when there's a likely problem.
    """
    import torch  # lazy: heavy import

    if not (is_rocm_torch() and torch.cuda.is_available()):
        return True, None

    try:
        props = torch.cuda.get_device_properties(0)
        device_name = props.name
        # gcnArchName looks like "gfx1151:sramecc+:xnack-"; keep the base arch.
        gfx = (getattr(props, "gcnArchName", "") or "").split(":")[0]
    except Exception:
        return True, None

    if not gfx:
        return True, None

    try:
        arch_list = [a.split(":")[0] for a in torch.cuda.get_arch_list()]
    except Exception:
        arch_list = []

    override = os.environ.get("HSA_OVERRIDE_GFX_VERSION")
    if arch_list and gfx not in arch_list and not override:
        return False, (
            f"{device_name} ({gfx}) is not in this PyTorch ROCm build's compiled "
            f"architectures ({', '.join(arch_list)}). Set HSA_OVERRIDE_GFX_VERSION "
            f"to a supported architecture, or install a ROCm PyTorch build that "
            f"targets {gfx} (see backend/requirements-rocm.txt)."
        )

    return True, None


def get_rocm_info() -> dict[str, object]:
    """Collect ROCm/AMD GPU details for diagnostics and the health endpoint.

    Returns:
        A mapping with ``is_rocm``, ``rocm_version``, ``device_name``,
        ``gfx_arch`` and ``hsa_override`` keys. Values are ``None`` when not
        applicable.
    """
    info: dict[str, object] = {
        "is_rocm": is_rocm_torch(),
        "rocm_version": None,
        "device_name": None,
        "gfx_arch": None,
        "hsa_override": os.environ.get("HSA_OVERRIDE_GFX_VERSION"),
    }

    version = get_rocm_version()
    if version is not None:
        info["rocm_version"] = f"{version[0]}.{version[1]}"

    if not info["is_rocm"]:
        # Fall back to the kernel-reported architecture even without ROCm torch,
        # so diagnostics can still surface "AMD GPU present, wrong PyTorch build".
        archs = detect_amd_gpu_archs()
        if archs:
            info["gfx_arch"] = archs[0]
        return info

    try:
        import torch  # lazy: heavy import

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["device_name"] = props.name
            gfx = (getattr(props, "gcnArchName", "") or "").split(":")[0]
            info["gfx_arch"] = gfx or None
    except Exception:
        pass

    if not info["gfx_arch"]:
        archs = detect_amd_gpu_archs()
        if archs:
            info["gfx_arch"] = archs[0]

    return info
