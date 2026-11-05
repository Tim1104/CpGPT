#!/bin/bash
# 用法: ./run.sh SAMPLE.csv [SAMPLE.csv ...]   (结果写到 systems_age_results.csv)
set -e
[ $# -ge 1 ] || { echo "用法: $0 SAMPLE.csv [SAMPLE.csv ...]"; exit 1; }
[ -f ~/sa_data/SystemsAge_data.qs2 ] || { echo "缺少 ~/sa_data/SystemsAge_data.qs2"; exit 1; }
D=~/sa_data; H=$(cd "$(dirname "$0")" && pwd)
mkdir -p /tmp/sa_in; rm -f /tmp/sa_in/*; for f in "$@"; do cp "$f" /tmp/sa_in/; done
docker run --rm -v "$D":/data:ro -v "$H":/code:ro -v /tmp/sa_in:/in:ro -v "$H":/out sa-r \
  Rscript /code/run_systems_age.R /data/SystemsAge_data.qs2 /out/systems_age_results.csv $(for f in "$@"; do echo /in/$(basename "$f"); done)
