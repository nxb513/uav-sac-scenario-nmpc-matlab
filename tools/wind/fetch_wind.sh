#!/usr/bin/env bash
# Download the public real-wind data from their original sources and build the
# force series in wind/series (see tools/wind/prepare_wind_series.py). The raw and
# derived data stay inside the CI job; they are never committed or uploaded.
set -euo pipefail
python3 -m pip install --quiet pandas numpy h5py
mkdir -p wind/nf wind/swuf
base=https://raw.githubusercontent.com/aerorobotics/neural-fly/main/data
for f in experiment/custom_figure8_baseline_nowind experiment/custom_figure8_baseline_35wind \
         experiment/custom_figure8_baseline_70wind experiment/custom_figure8_baseline_70p20sint \
         experiment/custom_figure8_baseline_100wind training/custom_random3_baseline_nowind \
         training/custom_random3_baseline_10wind training/custom_random3_baseline_20wind \
         training/custom_random3_baseline_30wind training/custom_random3_baseline_40wind \
         training/custom_random3_baseline_50wind; do
  curl -sSfL --retry 3 -o "wind/nf/$(basename "$f").csv" "$base/$f.csv"
done
curl -sSfL --retry 3 -o wind/swuf/2025.zip \
  "https://zenodo.org/api/records/17700905/files/2025_field_measurements.zip/content"
unzip -q -o wind/swuf/2025.zip -d wind/swuf
python3 tools/wind/prepare_wind_series.py --nf wind/nf --swuf wind/swuf --out wind/series
