# CycloneAI IBTrACS Explorer

A small Dash dashboard for exploring the NOAA/NCEI International Best Track
Archive for Climate Stewardship (IBTrACS). It shows storm counts, peak wind,
peak-intensity locations, storm counts by season, and storm counts by basin.

## 1. Get the dataset

The IBTrACS CSV is about 138 MB and is intentionally not stored in this Git
repository. Download the CSV from the
[NOAA/NCEI IBTrACS page](https://www.ncei.noaa.gov/products/international-best-track-archive)
or provide it to collaborators through an approved shared drive. Save it as:

```text
Data/ibtracs.csv
```

If you keep it elsewhere, set `IBTRACS_CSV_PATH` to its full path before
starting the app. In PowerShell:

```powershell
$env:IBTRACS_CSV_PATH = "C:\path\to\ibtracs.csv"
```

## 2. Install and run (Windows PowerShell)

From the project folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Open the local URL printed by Dash (normally
`http://127.0.0.1:8050`) in your browser. The first start reads the full CSV, so
it may take a little while.

The trained model is stored with Git LFS. Install Git LFS before cloning, then
run `git lfs pull` in the project folder to download the model.

## Sharing with collaborators

Push the source code to GitHub, but keep the large CSV out of ordinary Git
history. Each collaborator should download the same IBTrACS CSV or receive it
through a shared drive, then place it in `Data/ibtracs.csv` or configure
`IBTRACS_CSV_PATH` as described above. Git LFS is another option if you
intentionally want the dataset versioned and the repository's LFS quota allows
it.