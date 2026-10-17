"""Export the upstream finance example's full and debugging data without Docker."""

import argparse
import json
import os
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernels", type=int, default=8)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    os.chdir(root)
    provider = Path.home() / ".qlib/qlib_data/cn_data"
    if not (provider / "calendars/day.txt").is_file():
        raise FileNotFoundError("Download the official Qlib CN daily dataset before exporting factors.")
    full_dir = root / "git_ignore_folder/factor_implementation_source_data"
    debug_dir = root / "git_ignore_folder/factor_implementation_source_data_debug"
    if any((folder / "daily_pv.h5").exists() for folder in (full_dir, debug_dir)):
        raise FileExistsError("Factor exports already exist; inspect them before preparing again.")

    import qlib
    qlib.init(provider_uri=str(provider), kernels=args.kernels)
    from qlib.data import D

    # Same fields, date restriction and first 100 debug instruments as upstream
    # rdagent/scenarios/qlib/experiment/factor_data_template/generate.py.
    instruments = D.instruments()
    fields = ["$open", "$close", "$high", "$low", "$volume", "$factor"]
    data = D.features(instruments, fields, freq="day").swaplevel().sort_index().loc["2008-12-29":]
    full_dir.mkdir(parents=True, exist_ok=True)
    data.to_hdf(full_dir / "daily_pv.h5", key="data")
    debug = (
        D.features(instruments, fields, start_time="2018-01-01", end_time="2019-12-31", freq="day")
        .swaplevel().sort_index().swaplevel()
        .loc[data.reset_index()["instrument"].unique()[:100]]
        .swaplevel().sort_index()
    )
    debug_dir.mkdir(parents=True, exist_ok=True)
    debug.to_hdf(debug_dir / "daily_pv.h5", key="data")
    readme = root / "rdagent/scenarios/qlib/experiment/factor_data_template/README.md"
    for folder in (full_dir, debug_dir):
        shutil.copyfile(readme, folder / "README.md")
    result = {
        "full_shape": list(data.shape),
        "debug_shape": list(debug.shape),
        "full_date_min": str(data.index.get_level_values("datetime").min()),
        "full_date_max": str(data.index.get_level_values("datetime").max()),
        "full_instruments": len(data.index.get_level_values("instrument").unique()),
        "debug_instruments": len(debug.index.get_level_values("instrument").unique()),
        "fields": fields,
    }
    manifest = root / "git_ignore_folder/reproduction/factor-data-manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
