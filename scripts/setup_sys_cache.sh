# Relocate compiler / Ray / Python temp files onto autodl-tmp (data disk).
# Source from training launchers. Idempotent: existing symlinks are left alone.
SYS_CACHE_ROOT="${SYS_CACHE_ROOT:-/root/autodl-tmp/sys-cache}"
mkdir -p \
  "${SYS_CACHE_ROOT}/torchinductor" \
  "${SYS_CACHE_ROOT}/triton/cache" \
  "${SYS_CACHE_ROOT}/tmp" \
  "${SYS_CACHE_ROOT}/ray" \
  "${SYS_CACHE_ROOT}/cuda" \
  "${SYS_CACHE_ROOT}/xdg"

_shopsimrl_relocate_dir() {
  local src="$1" dest="$2"
  mkdir -p "$(dirname "${src}")" "${dest}"
  if [[ -L "${src}" ]]; then
    return 0
  fi
  if [[ -d "${src}" ]]; then
    rsync -a "${src}/" "${dest}/"
    rm -rf "${src}"
  fi
  ln -sfn "${dest}" "${src}"
}

_shopsimrl_relocate_dir /tmp/torchinductor_root "${SYS_CACHE_ROOT}/torchinductor"
_shopsimrl_relocate_dir /root/.triton "${SYS_CACHE_ROOT}/triton"
unset -f _shopsimrl_relocate_dir

export TORCHINDUCTOR_CACHE_DIR="${SYS_CACHE_ROOT}/torchinductor"
export TRITON_CACHE_DIR="${SYS_CACHE_ROOT}/triton/cache"
export CUDA_CACHE_PATH="${SYS_CACHE_ROOT}/cuda"
export TMPDIR="${SYS_CACHE_ROOT}/tmp"
export TEMP="${SYS_CACHE_ROOT}/tmp"
export TMP="${SYS_CACHE_ROOT}/tmp"
export RAY_TMPDIR="${SYS_CACHE_ROOT}/ray"
export XDG_CACHE_HOME="${SYS_CACHE_ROOT}/xdg"
