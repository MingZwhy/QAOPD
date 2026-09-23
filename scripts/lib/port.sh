#!/usr/bin/env bash
# A rendezvous port for `accelerate launch` that nothing else already holds.
#
# Why this exists: every evaluation entry point used to derive its port from a
# constant or from the GPU index, which collides in two ways. Two evaluations of
# the same kind on one host pick the same constant; and an evaluation sharing a
# host with training collides with a Ray worker, which takes arbitrary high
# ports. Either way accelerate dies with EADDRINUSE, or hangs for 900 s and then
# reports DistNetworkError -- and it does so *after* the export has been paid
# for, which on a 4B student is several minutes of GPU time thrown away.
#
# Binding to port 0 makes the kernel name a port that is free at that moment,
# which is the only thing a userspace check can honestly claim. `ss` is not an
# option: it is absent from the training images, and a missing binary makes a
# port check silently report "nothing is listening".
#
# Call it as late as possible -- immediately before the launch, not before an
# export -- because the guarantee expires the moment the probe socket closes.
#
# Usage: source "$QAOPD_ROOT/scripts/lib/port.sh"; PORT=$(free_port)
# Reads: PY (optional) -- the interpreter to probe with; falls back to python3.

free_port() {
    local py="${PY:-python3}"
    command -v "$py" >/dev/null 2>&1 || py=python3
    "$py" - <<'PY' 2>/dev/null || return 1
import socket

with socket.socket() as s:
    s.bind(("", 0))
    print(s.getsockname()[1])
PY
}
