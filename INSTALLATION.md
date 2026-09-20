# Installation Guide

## Requirements

- Python 3.10–3.12
- Windows 10/11, Ubuntu 20.04+, or macOS 12+
- 4 GB RAM minimum, 8 GB recommended
- ~500 MB disk space

## Step 1: Clone or Extract

```bash
git clone <repo-url>
cd apvt_project
```

Or extract the ZIP and open the folder.

## Step 2: Create Virtual Environment

```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux/macOS
python3 -m venv .venv
source .venv/bin/activate
```

If PowerShell blocks script execution:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```

## Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Step 4: Verify

```bash
python src/test_complete.py
```

Expected: all checks pass with ✅.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `AquaCrop` import fails | Empirical crop model is used automatically — no action needed |
| `bifacial_radiance` missing | Ray-tracing validation is skipped; view-factor model still works |
| TMY file not found | Download from PVGIS (see FIRST_RUN.md) and place in `data/` |
| Memory errors on low-spec | Reduce `n_ground_segments` in `config.py` to 50 |

## Key Dependencies

| Package | Purpose |
|---------|---------|
| numpy, pandas, scipy | Core numerics |
| pvlib | Solar position, irradiance transposition |
| matplotlib, seaborn | Plotting |
| aquacrop | Crop model (optional fallback included) |
| bifacial-radiance | 3D ray-tracing (optional) |
