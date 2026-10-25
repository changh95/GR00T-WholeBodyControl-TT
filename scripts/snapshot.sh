#!/usr/bin/env bash
# Frozen copy of a commit for a device session: snapshots/NAME is a detached git worktree of COMMIT (default HEAD),
# so `git worktree list` records which commit every hold ran. Device sessions run from a snapshot, never from the
# working tree, so edits made during a long run cannot change the code mid-run.
#
#   scripts/snapshot.sh m3f [COMMIT]   then   source snapshots/m3f/env.sh && $DEVX $RT/scripts/chip_session.sh ...
#
# The worktree gets what git does not track but a session needs, copied from the main tree: generated/ (tt-metal
# kernel cache), x280/fw/build/ (firmware images; checked to come from COMMIT's sources), x280/host/libwcio.so,
# x280/sim/spike-install/. runs/ and .venv-x280/ are links back to the main tree, so every session's data lands in
# one runs/. Remove one with `git worktree remove --force snapshots/NAME`.
set -euo pipefail
TOP=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
name=${1:?usage: scripts/snapshot.sh NAME [COMMIT]}
commit=$(git -C "$TOP" rev-parse --verify "${2:-HEAD}^{commit}")
dst=$TOP/snapshots/$name
[ ! -e "$dst" ] || { echo "snapshot: $dst exists" >&2; exit 2; }
if [ "${2:-HEAD}" = HEAD ] && [ -n "$(git -C "$TOP" status --porcelain --untracked-files=no)" ]; then
    echo "snapshot: the working tree has uncommitted changes; commit them or name a COMMIT" >&2
    exit 2
fi
if ! git -C "$TOP" diff --quiet "$commit" -- x280/fw; then
    echo "snapshot: x280/fw differs from $commit, so x280/fw/build may not be its image; build from $commit" >&2
    exit 2
fi
git -C "$TOP" worktree add --detach "$dst" "$commit" >/dev/null
for p in generated x280/fw/build x280/host/libwcio.so x280/sim/spike-install; do
    [ -e "$TOP/$p" ] || continue
    mkdir -p "$(dirname "$dst/$p")"
    cp -a "$TOP/$p" "$dst/$p"
done
for l in runs .venv-x280; do
    [ -e "$TOP/$l" ] && ln -s "$TOP/$l" "$dst/$l"
done
echo "$commit" >"$dst/SNAPSHOT_COMMIT"
echo "snapshot $name: $dst at $(git -C "$TOP" log -1 --format='%h %s' "$commit")"
