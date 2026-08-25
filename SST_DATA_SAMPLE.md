# Getting a training-ready sample of the SST-TG datasets

This documents how to pull a small, tractable sample of the published
"Decaying Stably-Stratified Turbulence, Taylor-Green initialized" (SST-TG)
DNS datasets from OLCF's public Globus collection, land it under
`$PROJWORK/stf249`, and turn it into an ML-ready `.npz` using the
`dataloaders/sst-binary.py` loader already in this repo. It also gives a
data-catalog entry template with real metadata for each of the three
published datasets.

**If the raw data is already sitting on disk** (e.g. `P1F4R3200` under
`/opt/data/sickle/P1F4R32_nx512ny512nz256_6vars/`), skip Steps 1-3 below and
just run sickle's own hypercube/maxent subsampler directly against it --
`config/SST/P1/test.yaml` already points at that exact path:

```bash
python subsample.py config/SST/P1/test.yaml --plot
```

See [EXAMPLES.md](./EXAMPLES.md#visualizing-subsampled-hypercubes-and-points)
for what `--plot` produces and how to visualize the sampled hypercubes/points.
The rest of this document (`scripts/prepare_sst_sample.py`) is a separate,
simpler path for turning a small strided crop into a raw `.npz` -- without
sickle's hypercube extraction or maxent subsampling -- useful mainly right
after a fresh Globus transfer of just a few files.

## The three published datasets

All three are DOE/OLCF-published DNS datasets from the same authors (de Bruyn
Kops, Riley, Couchman, Gopalakrishnan Meena), same physical setup (Taylor-Green
initial condition, Froude number Fr=4, Reynolds number Re=3200), differing
only in Prandtl number and grid resolution. Each snapshot has 4 variables:
velocity components `u`, `v`, `w` and perturbed density `r` (rho), each stored
as its own 32-bit little-endian binary file.

| Dataset | DOI | Grid (nx x ny x nz) | Snapshots | Size/variable/snapshot | Total size |
|---|---|---|---|---|---|
| SST-TG-P1F4R3200  | [10.13139/OLCF/2530508](https://doi.org/10.13139/OLCF/2530508) | 512 x 512 x 256     | 15,000 | 255 MB | 15.3 TB |
| SST-TG-P7F4R3200  | [10.13139/OLCF/2566733](https://doi.org/10.13139/OLCF/2566733) | 1280 x 1280 x 640   | 15,250 | 4 GB   | 244 TB  |
| SST-TG-P50F4R3200 | [10.13139/OLCF/2566017](https://doi.org/10.13139/OLCF/2566017) | 3584 x 3584 x 1792  | 1,680  | 85.8 GB| 577 TB  |

**Recommendation: start with P1F4R3200.** It is 336x smaller per snapshot
than P50 and the default grid dimensions already hard-coded in this repo's
`args.py` (`--nx 512+2 --ny 512 --nz 256`) match its resolution almost
exactly, meaning `dataloaders/sst-binary.py` is already effectively tuned for
this dataset without changing a flag.

All three are served from the **same public Globus endpoint** (a "world-shared"
OLCF DOI-data collection; no OLCF account or DOE credentials needed to *read*
it, only a free Globus login to drive a transfer):

- **Source Globus endpoint UUID:** `57618e0a-2c99-45ff-9694-24141b92fa17`
- **Source paths:**
  - P1F4R3200:  `/gen101/world-shared/doi-data/OLCF/202504/10.13139_OLCF_2530508/`
  - P7F4R3200:  `/gen101/world-shared/doi-data/OLCF/202506/10.13139_OLCF_2566733/`
  - P50F4R3200: `/gen101/world-shared/doi-data/OLCF/202506/10.13139_OLCF_2566017/`

Each dataset's own path has a `README` describing the exact file naming
convention; **confirm it before trusting the guessed pattern below.**

## Step 0: find your destination collection

You need the Globus UUID of *your own* destination collection (your OLCF DTN
/ `$PROJWORK` access point). This is specific to your account/allocation and
not something to guess -- look it up yourself:

```bash
module load globus-cli   # or: pip install globus-cli --user
globus login
globus endpoint search "OLCF DTN"          # find your destination collection
globus endpoint search "world-shared doi"  # sanity-check the source collection above
```

Set it as a shell variable for the rest of this doc:

```bash
export SRC=57618e0a-2c99-45ff-9694-24141b92fa17
export DST=<your-olcf-dtn-collection-uuid>
export SAMPLE_DIR=$PROJWORK/stf249/sst_turbulence/raw/P1F4R3200
mkdir -p "$SAMPLE_DIR"
```

## Step 1: list the source directory before transferring anything

Do not guess filenames for a real transfer -- list them first:

```bash
globus ls $SRC:/gen101/world-shared/doi-data/OLCF/202504/10.13139_OLCF_2530508/
```

Based on this repo's own loader (`dataloaders/sst-binary.py`'s
`_extract_times` regex `r'_([0-9]+\.[0-9]+)$'` and its
`file_path = os.path.join(path, f'{var}_{ts:0.6f}')` construction), the
expected naming pattern is:

```
u_<timestep:.6f>
v_<timestep:.6f>
w_<timestep:.6f>
r_<timestep:.6f>
```

e.g. `u_0.010000`, `v_0.010000`, `w_0.010000`, `r_0.010000` for the snapshot at
t=0.01. **Confirm the actual names against the `globus ls` output and the
dataset's own README before proceeding** -- the exact timestep values and
zero-padding are set by the simulation's actual output cadence, not by this
repo's defaults.

## Step 2: transfer a small sample (a handful of timesteps, all 4 variables)

Once you have confirmed real filenames from Step 1, build a batch file listing
just the timesteps you want (start with 2-3 timesteps x 4 variables = 8-12
files x 255 MB ~= 2-3 GB for P1F4R3200, a genuinely small sample):

```bash
cat > /tmp/sst_sample_batch.txt <<EOF
u_0.010000 u_0.010000
v_0.010000 v_0.010000
w_0.010000 w_0.010000
r_0.010000 r_0.010000
u_0.020000 u_0.020000
v_0.020000 v_0.020000
w_0.020000 w_0.020000
r_0.020000 r_0.020000
EOF

globus transfer $SRC:/gen101/world-shared/doi-data/OLCF/202504/10.13139_OLCF_2530508/ \
                $DST:$SAMPLE_DIR/ \
                --batch /tmp/sst_sample_batch.txt \
                --label "SST-TG-P1F4R3200 sample (2 timesteps)"
```

`globus transfer` returns a task ID immediately; the transfer runs
asynchronously. Check status with:

```bash
globus task show <task-id>
```

For a first smoke test before committing to even that, you can transfer a
single file (~255 MB) by dropping the `--batch` flag and passing one
`--path`-style source/dest file pair instead; see `globus transfer --help`.

## Step 3: make it training-ready

Once the raw sample lands under `$SAMPLE_DIR`, use this repo's existing
loader to read a strided sub-cube (no need to read the full grid into memory)
and save it as a compact `.npz`. A ready-to-run wrapper is provided at
`scripts/prepare_sst_sample.py`:

```bash
cd ~/turbulence/sickle
python scripts/prepare_sst_sample.py \
    --path "$SAMPLE_DIR" \
    --timesteps 0.01 0.02 \
    --nx 514 --ny 512 --nz 256 \
    --nxsl 128 --nysl 128 --nzsl 128 \
    --nxskip 4 --nyskip 4 --nzskip 2 \
    --out "$PROJWORK/stf249/sst_turbulence/processed/P1F4R3200_sample.npz"
```

This strides a 128x128x128 sub-cube out of the full 512x512x256 grid (4x/4x/2x
downsampling) for each requested timestep and variable, and writes one `.npz`
containing `X` (shape `[T, nx_sl, ny_sl, nz_sl, 4]`, channel order
`[u, v, w, r]`), the 1-D grid coordinate arrays `x`, `y`, `z`, and the list of
timesteps actually loaded. Adjust `--nxsl/--nysl/--nzsl` and the skip factors
to change the sample's resolution/size; `--gravity` defaults to `z` (matching
this repo's default) and should match the dataset's actual stratification
axis (check the dataset's README/paper).

## Step 4: record it in your data catalog

A template entry per dataset (adapt field names to whatever catalog schema
your team actually uses -- this is a generic, tool-agnostic structure):

```yaml
- name: SST-TG-P1F4R3200
  doi: 10.13139/OLCF/2530508
  url: https://doi.org/10.13139/OLCF/2530508
  description: >
    Direct numerical simulation of decaying stably-stratified turbulence,
    Taylor-Green vortex initial condition, Pr=1, Fr=4, Re=3200.
  authors:
    - Stephen M. de Bruyn Kops (UMass Amherst)
    - James J. Riley (University of Washington)
    - Miles M. P. Couchman (York University)
    - Muralikrishnan Gopalakrishnan Meena (ORNL)
  grid: [512, 512, 256]
  variables: [u, v, w, r]
  num_snapshots: 15000
  size_per_variable_per_snapshot: 255MB
  total_size: 15.3TB
  format: binary, 32-bit little-endian
  source:
    globus_endpoint: 57618e0a-2c99-45ff-9694-24141b92fa17
    globus_path: /gen101/world-shared/doi-data/OLCF/202504/10.13139_OLCF_2530508/
  sample_path: $PROJWORK/stf249/sst_turbulence/raw/P1F4R3200/     # fill in after Step 2
  processed_path: $PROJWORK/stf249/sst_turbulence/processed/P1F4R3200_sample.npz  # fill in after Step 3
  license: not specified on landing page as of dataset release
  contact: see DOI landing page (doi.ccs.ornl.gov) for current contact info
```

Repeat the same structure for `SST-TG-P7F4R3200` (DOI `10.13139/OLCF/2566733`,
grid `[1280, 1280, 640]`, `15250` snapshots, `4GB`/var/snapshot, `244TB` total)
and `SST-TG-P50F4R3200` (DOI `10.13139/OLCF/2566017`, grid
`[3584, 3584, 1792]`, `1680` snapshots, `85.8GB`/var/snapshot, `577TB` total),
both using the same Globus endpoint UUID and their own paths listed in the
table above.

## Notes / caveats

- I have not run this download myself: the actual transfer requires your own
  Globus identity and OLCF `$PROJWORK` access, neither of which is available
  from this environment. Everything above is a verified-facts recipe (dataset
  metadata pulled directly from the DOI landing pages on 2026-08-21), not a
  completed run.
- `scripts/prepare_sst_sample.py` (Step 3) *is* tested in this environment,
  but only against synthetic dummy binary files matching the expected naming
  convention and grid shape -- not the real dataset, since the real data
  isn't downloadable from here. Re-verify shapes/byte order against a real
  transferred file before trusting output at scale.
- The exact file-naming timestep values/padding for a given dataset are set
  by that simulation's own output cadence; do not assume `0.010000`/`0.020000`
  exist verbatim until you've run the `globus ls` in Step 1.
