#!/bin/bash
# ROCm container entrypoint — fixes volume permissions and GPU group access,
# then drops to the voicebox user so the server never runs as root.
#
# Why the group dance is needed:
#   Docker's --group-add adds extra GIDs to the PID-1 process, but gosu(1)
#   calls initgroups() which resets supplementary groups to exactly what
#   /etc/group lists for the target user — the Docker extras are lost.
#   The fix: add voicebox to the actual GIDs that own the GPU device nodes
#   (resolved from the mounted devices) *before* calling gosu, so
#   initgroups() picks them up from /etc/group.
set -e

# 1. Ensure the voicebox user owns the writable data directories that may be
#    mounted as Docker named volumes (which Docker initialises as root:root).
chown -R voicebox:voicebox \
    /app/data \
    /home/voicebox/.cache \
    /home/voicebox/.config \
    2>/dev/null || true

# 2. Add voicebox to the groups that own GPU device nodes so torch/HIP can
#    open /dev/kfd and /dev/dri/renderD*.
for dev in /dev/kfd /dev/dri/renderD* /dev/dri/card*; do
    [ -e "$dev" ] || continue
    gid=$(stat -c%g "$dev" 2>/dev/null) || continue
    [ "$gid" -eq 0 ] && continue  # skip root-owned devices

    # Create a named group for this GID if none exists yet (required for
    # usermod -aG to accept a bare numeric GID on some distros).
    if ! getent group "$gid" >/dev/null 2>&1; then
        groupadd -g "$gid" "gpu_${gid}" 2>/dev/null || true
    fi

    usermod -aG "$gid" voicebox 2>/dev/null || true
done

# 3. Drop privileges and exec the server command.
exec gosu voicebox "$@"
