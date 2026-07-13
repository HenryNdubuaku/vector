#!/bin/sh
set -eu

SUDO=""
[ "$(id -u)" = "0" ] || SUDO="sudo"

if [ "$(uname -s)" = "Darwin" ]; then
    command -v brew >/dev/null || { echo "vector needs homebrew on macos: https://brew.sh" >&2; exit 1; }
    command -v protoc >/dev/null || brew install protobuf
else
    $SUDO env DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=l apt-get update
    $SUDO env DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=l apt-get install -y libclang-dev unzip curl git build-essential ca-certificates
    need_protoc=1
    if command -v protoc >/dev/null; then
        case "$(protoc --version | cut -d' ' -f2)" in
            3.*) need_protoc=1 ;;
            *) need_protoc=0 ;;
        esac
    fi
    if [ "$need_protoc" = "1" ]; then
        arch="x86_64"
        [ "$(uname -m)" = "aarch64" ] && arch="aarch_64"
        curl -fsSLO "https://github.com/protocolbuffers/protobuf/releases/download/v25.3/protoc-25.3-linux-$arch.zip"
        $SUDO unzip -o "protoc-25.3-linux-$arch.zip" -d /usr/local bin/protoc 'include/*' >/dev/null
        rm "protoc-25.3-linux-$arch.zip"
    fi
fi

if ! command -v cargo >/dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
fi
PATH="$HOME/.cargo/bin:$PATH"

if [ -d "$HOME/vector/.git" ]; then
    git -C "$HOME/vector" pull
else
    git clone https://github.com/HenryNdubuaku/vector.git "$HOME/vector"
fi
cargo install --path "$HOME/vector"
vector setup

echo
echo 'vector is installed; if the command is not found, run: . "$HOME/.cargo/env"'
