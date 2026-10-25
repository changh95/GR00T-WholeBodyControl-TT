#!/usr/bin/env bash
# Build spike (riscv-isa-sim) and dtc from source in user space (PLAN.md WP-3.2b; no sudo, none is installed).
#
# Why dtc: spike's configure refuses to run without `dtc` on PATH, and at run time spike execlp()s
# "dtc" to turn its generated DTS into the DTB it places in memory, unless --dtb is given
# (riscv-isa-sim configure.ac:63-65, riscv/dts.cc dtc_compile). dtc in turn needs flex and bison
# at build time, which are not installed either, so both are built from release tarballs into
# build/hosttools (build-time only; dtc does not need them at run time).
#
# Layout (all under this directory, see .gitignore):
#   build/{riscv-isa-sim,dtc,bison-*,flex-*}  sources        build/spike-build  spike out-of-tree build
#   build/hosttools                           flex + bison   spike-install      bin/{spike,dtc,...}
# SHAS.txt next to this script records what was actually built (tracked).
#
# Idempotent: every step writes a stamp keyed by its pinned source id and flags and is skipped when
# the stamp and its output exist. A source tree with local edits is never checked out over.
# Pinned to X280_CPUS (default 0,1,8-11,16,17,24-27) at nice 19 with -j8: off the real-time cores 2,3,4,6,7,12-15
# and their SMT siblings.
set -euo pipefail

SIM=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BUILD=$SIM/build
PREFIX=$SIM/spike-install
HOST=$BUILD/hosttools
JOBS=${JOBS:-8}
PIN=(nice -n 19 taskset -c "${X280_CPUS:-0,1,8-11,16,17,24-27}")

SPIKE_URL=https://github.com/riscv-software-src/riscv-isa-sim.git
SPIKE_SHA=${SPIKE_SHA:-2c300cd90eb74bb79f3fcdd58596a6f8d842d01c}   # master, 2026-10-21 (PR #2486)
DTC_URL=https://git.kernel.org/pub/scm/utils/dtc/dtc.git
DTC_SHA=${DTC_SHA:-8f48565e5cfedc74d3f7512f1e0188e9d85dc1de}       # tag v1.8.1
BISON_VER=3.8.2
BISON_URL=https://ftp.gnu.org/gnu/bison/bison-$BISON_VER.tar.xz
BISON_SHA256=9bba0214ccf7f1079c5d59210045227bcf619519840ebfa80cd3849cff5a5bf2
FLEX_VER=2.6.4
FLEX_URL=https://github.com/westes/flex/releases/download/v$FLEX_VER/flex-$FLEX_VER.tar.gz
FLEX_SHA256=e87aae032bf07c26f85ac0ed3250998c37621d95f8bd748b31f15b33c45ee995
SPIKE_CONFIGURE=(--prefix="$PREFIX" --with-boost=no --with-boost-asio=no --with-boost-regex=no)

log() { printf '[build_spike] %s\n' "$*" >&2; }

stamp_ok() {  # stamp_ok NAME KEY OUTPUT: true when NAME was built from KEY and OUTPUT exists
    [[ -f "$BUILD/.stamp-$1" && "$(cat "$BUILD/.stamp-$1")" == "$2" && -e "$3" ]]
}
stamp_set() { printf '%s' "$2" > "$BUILD/.stamp-$1"; }

git_at() {  # git_at URL DIR SHA: clone if missing, then detach at SHA; refuse to touch a dirty tree
    local url=$1 dir=$2 sha=$3
    if [[ ! -d "$dir/.git" ]]; then
        log "clone $url"
        git clone -q "$url" "$dir"
    fi
    if [[ "$(git -C "$dir" rev-parse HEAD)" != "$sha" ]]; then
        if [[ -n "$(git -C "$dir" status --porcelain --untracked-files=no)" ]]; then
            log "ERROR: $dir has local edits; not checking out $sha"; exit 2
        fi
        git -C "$dir" cat-file -e "$sha^{commit}" 2>/dev/null || git -C "$dir" fetch -q origin
        git -C "$dir" checkout -q --detach "$sha"
    fi
    [[ "$(git -C "$dir" rev-parse HEAD)" == "$sha" ]] || { log "ERROR: $dir not at $sha"; exit 2; }
    # spike vendors berkeley-softfloat-3 as a submodule (riscv-isa-sim .gitmodules); a no-op for dtc.
    git -C "$dir" submodule -q update --init --recursive
}

tarball() {  # tarball URL SHA256 DIR: download once, verify, unpack into DIR (its top-level dir)
    local url=$1 sum=$2 dir=$3 f=$BUILD/dl/${1##*/}
    mkdir -p "$BUILD/dl"
    [[ -f "$f" ]] || { log "download $url"; curl -fsSL -o "$f.part" "$url"; mv "$f.part" "$f"; }
    echo "$sum  $f" | sha256sum -c --quiet - || { log "ERROR: sha256 mismatch for $f"; exit 2; }
    [[ -d "$dir" ]] || tar -C "$BUILD" -xf "$f"
}

autotools_host() {  # autotools_host DIR: configure/make/install into hosttools
    local dir=$1
    (cd "$dir" && "${PIN[@]}" ./configure -q --prefix="$HOST" --disable-nls >/dev/null \
        && "${PIN[@]}" make -s -j"$JOBS" >/dev/null && make -s install >/dev/null)
}

mkdir -p "$BUILD" "$PREFIX/bin"
export PATH="$PREFIX/bin:$HOST/bin:$PATH"

# 1. bison + flex (build-time tools of dtc).
tarball "$BISON_URL" "$BISON_SHA256" "$BUILD/bison-$BISON_VER"
if ! stamp_ok bison "$BISON_SHA256" "$HOST/bin/bison"; then
    log "build bison $BISON_VER"; autotools_host "$BUILD/bison-$BISON_VER"; stamp_set bison "$BISON_SHA256"
fi
tarball "$FLEX_URL" "$FLEX_SHA256" "$BUILD/flex-$FLEX_VER"
if ! stamp_ok flex "$FLEX_SHA256" "$HOST/bin/flex"; then
    log "build flex $FLEX_VER"; autotools_host "$BUILD/flex-$FLEX_VER"; stamp_set flex "$FLEX_SHA256"
fi

# 2. dtc (run-time dependency of spike when no --dtb is passed). Built with dtc's Makefile: upstream prefers
#    meson, which is not installed; the Makefile still builds v1.8.1 (it only prints a deprecation warning).
git_at "$DTC_URL" "$BUILD/dtc" "$DTC_SHA"
DTC_FLAGS=(PREFIX="$PREFIX" NO_PYTHON=1 NO_YAML=1 NO_VALGRIND=1)
if ! stamp_ok dtc "$DTC_SHA ${DTC_FLAGS[*]}" "$PREFIX/bin/dtc"; then
    log "build dtc $DTC_SHA"
    "${PIN[@]}" make -s -C "$BUILD/dtc" -j"$JOBS" "${DTC_FLAGS[@]}" dtc >/dev/null
    install -m 0755 "$BUILD/dtc/dtc" "$PREFIX/bin/dtc"
    stamp_set dtc "$DTC_SHA ${DTC_FLAGS[*]}"
fi

# 3. spike.
git_at "$SPIKE_URL" "$BUILD/riscv-isa-sim" "$SPIKE_SHA"
if ! stamp_ok spike "$SPIKE_SHA ${SPIKE_CONFIGURE[*]}" "$PREFIX/bin/spike"; then
    log "build spike $SPIKE_SHA (takes a few minutes)"
    rm -rf "$BUILD/spike-build" "$BUILD/.stamp-slim"; mkdir -p "$BUILD/spike-build"
    (cd "$BUILD/spike-build" \
        && "${PIN[@]}" "$BUILD/riscv-isa-sim/configure" "${SPIKE_CONFIGURE[@]}" >configure.out \
        && "${PIN[@]}" make -j"$JOBS" >make.out 2>&1 \
        && "${PIN[@]}" make install >install.out 2>&1) || { log "ERROR: spike build failed, see $BUILD/spike-build/*.out"; exit 1; }
    stamp_set spike "$SPIKE_SHA ${SPIKE_CONFIGURE[*]}"
fi
# The -g -O2 build leaves ~5 GB of objects/executables and a ~1.8 GB debug-laden install on a 92 %-full disk:
# strip debug info from the install (to ~86 MB) and delete the build outputs (configure/make logs stay).
# KEEP_DEBUG=1 skips this.
if [[ "${KEEP_DEBUG:-0}" != 1 ]] && ! stamp_ok slim "$SPIKE_SHA" "$PREFIX/bin/spike"; then
    log "slim: strip debug info in $PREFIX, delete the spike object files"
    find "$PREFIX/bin" "$PREFIX/lib" -type f \( -perm -u+x -o -name '*.a' -o -name '*.so*' \) \
        ! -name dtc -exec strip --strip-debug {} + 2>/dev/null || true
    find "$BUILD/spike-build" -type f \( -name '*.o' -o -name '*.a' -o -name '*.so' -o -name '*.gch' \
        -o -name '*.d' -o -perm -u+x \) ! -name configure ! -name config.status -delete 2>/dev/null || true
    stamp_set slim "$SPIKE_SHA"
fi

# 4. Smoke check and the record of what was built.
"$PREFIX/bin/dtc" --version >/dev/null
"$PREFIX/bin/spike" --help >/dev/null 2>&1 || [[ $? -eq 1 ]]   # spike --help exits 1 by design
{
    echo "# Written by build_spike.sh; what $PREFIX was built from."
    echo "riscv-isa-sim $SPIKE_SHA $SPIKE_URL"
    git -C "$BUILD/riscv-isa-sim" submodule status --recursive | sed 's/^[ +-]/  submodule /'
    echo "dtc $DTC_SHA $DTC_URL ($(git -C "$BUILD/dtc" describe --tags 2>/dev/null || echo '?'))"
    echo "bison-$BISON_VER sha256:$BISON_SHA256 $BISON_URL"
    echo "flex-$FLEX_VER sha256:$FLEX_SHA256 $FLEX_URL"
    echo "spike configure: ${SPIKE_CONFIGURE[*]}" | sed "s|$PREFIX|<prefix>|"
    echo "dtc make flags: ${DTC_FLAGS[*]}" | sed "s|$PREFIX|<prefix>|"
    echo "host compiler: $(gcc --version | head -1)"
} > "$SIM/SHAS.txt"
log "ok: $PREFIX/bin/spike, $PREFIX/bin/dtc (record in $SIM/SHAS.txt)"
