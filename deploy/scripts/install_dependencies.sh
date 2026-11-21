#!/bin/bash
set -Eeuo pipefail

if command -v docker >/dev/null 2>&1; then
  systemctl enable --now docker || true
  exit 0
fi

if command -v dnf >/dev/null 2>&1; then
  dnf install -y docker
elif command -v yum >/dev/null 2>&1; then
  yum install -y docker
elif command -v apt-get >/dev/null 2>&1; then
  apt-get update
  apt-get install -y docker.io
else
  echo "Unsupported Linux package manager; install Docker on the EC2 instance." >&2
  exit 1
fi

systemctl enable --now docker
