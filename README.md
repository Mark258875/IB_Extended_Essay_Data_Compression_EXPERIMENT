# Brotli & LZ4 EE Mini‑Bench (Python)

Benchmark **Brotli** and **LZ4** lossless compression on your datasets or controlled synthetic data. The suite measures:

* Compression ratio & **bits per byte (bpb)**
* Empirical **zero‑order entropy H₁** (bpb)
* **% over entropy** (how far the result is from H₁)
* Compression / decompression **throughput** (MB/s and MiB/s)
* Reproducibility **meta** (CPU, cores, power plan, library versions, repeats, warmup, etc.)

> Key idea: compressibility depends not only on the **amount** of each symbol but also on their **arrangement** (dependencies/runs). We test both marginal distributions and order effects.

---

## Repo layout

```
.
├── bench_compress.py        # Core timing + metrics (shared by all sweeps)
├── config.py                # YAML config loader & validation
├── conf/
│   └── config.yaml          # Your experiment configuration
├── entropy_scan.py          # Computes H₁ (bpb) + optional bpc for text
├── make_binary_data.py      # Generates binary sources (ascii01 / bitpack)
├── runtime_tuning.py        # CPU affinity + priority controls
├── sweep_brotli.py          # Sweep Brotli quality 0..11 (no CLI)
├── sweep_lz4.py             # Sweep LZ4 level 0..16 (no CLI)
└── results/                 # CSVs + meta will be written here
```

---

## Install

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

**requirements.txt** should include at least:

```
brotli>=1.1.0
lz4>=4.3.2
PyYAML>=6.0.1
psutil>=5.9.8
py-cpuinfo>=9.0.0
```

---

## Configure (conf/config.yaml)

```yaml
paths:
  input_globs:
    - "./data/*.txt"
    - "./data/binary/**/*.txt"   # ascii01
    - "./data/binary/**/*.bin"   # bitpack

  output_dir: "./results"
  # Per‑algorithm outputs
  output_brotli: "results_brotli.csv"
  output_lz4:    "results_lz4.csv"

runtime:
  cpu_affinity: 0      # Pin the process to a single core (index)
  priority: high       # low | normal | high | realtime
  warn_small_bytes: 10000000   # Warn if file < 10 MB (throughput gets noisy)

brotli:
  quality: 2           # default for single‑run tools (sweeps override)
  mode: text           # generic | text | font
  lgwin: null          # 10..24 or null for default

timing:
  repeats: 5           # median over N repeats
  warmup: 1            # warm‑up runs before timing

lz4:
  compression_level: 1 # default for single‑run tools (sweeps override)

synthetic:
  kind: zipf
  size: 10000000
  params:
    s: 1.2
    mu: 128
    sigma: 40
  seed: 123
```

> **Glob note:** the sweep scripts expand patterns **recursively**, so `**` works (e.g., `./data/binary/**/*.bin`).

---

## Generate controlled binary test data (optional but recommended)

This produces **binary sources with two symbols (0/1)** under different probabilities and orders, in two representations:

* `ascii01`: bytes `"0"`/`"1"` (human‑readable), size = *N bytes → N bits*.
* `bitpack`: bits packed MSB→LSB (binary), size = *N bytes → 8N bits*.

Run:

```bash
python make_binary_data.py
```

Outputs under `data/binary/` and creates a `manifest.csv` with `p`, order, representation, and file paths.

Parameters inside the script:

* `P_VALUES = [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 0.90, 0.95, 0.99]`
* Two sequential orders per `p`:

  * `blocks_zeros_then_ones`
  * `alternating` (0101… until one symbol runs out; remainder appended)
* `SIZE_BYTES = 10_000_000` per file (adjust as needed).

---

## Run the sweeps

Brotli (qualities 0..11):

```bash
python sweep_brotli.py
```

LZ4 (levels 0..16):

```bash
python sweep_lz4.py
```

Both scripts:

* Apply **runtime controls** (affinity, priority) from the YAML.
* Read inputs into memory before timing to avoid I/O bias.
* Use warm‑up + repeated timings; record **medians**.
* Append results to per‑algo CSVs in `results/` and write a `*.meta.json` with environment details.

---

## Compute entropy (H₁) for inputs

```bash
python entropy_scan.py
```

Writes `results/entropy.csv` with:

* `file, size_bytes, unique_bytes, entropy_bpb, max_bits_bytes, eff_vs_8bit, decoded_utf8, unique_chars, entropy_bpc, max_bits_chars, eff_vs_max_chars`

* **entropy\_bpb**: empirical H₁ over **bytes** (0..255), in bits/byte.

* **entropy\_bpc**: H₁ over **characters** (after UTF‑8 decode), in bits/character (blank if not valid UTF‑8).

> Zeros don’t contribute: symbols with `p=0` add 0 to $-\sum p \log_2 p$. Using the 256‑byte alphabet is correct for byte‑oriented compressors.

---

## Output files

* `results/results_brotli.csv` — one row per (file, quality)
* `results/results_lz4.csv` — one row per (file, level)
* `results/*.meta.json` — environment + library versions + runtime controls used
* `results/entropy.csv` — entropy scan of inputs
* `data/binary/manifest.csv` — generated binary sources manifest

Each bench row includes:

* `dataset, alg, params, original_bytes, compressed_bytes, ratio, bpb`
* `entropy_bpb, %_over_entropy`
* `comp_time_s (p10/p90 optional), comp_MB_s, comp_MiB_s`
* `decomp_time_s (p10/p90 optional), decomp_MB_s, decomp_MiB_s`
* `repeats, env_python, env_platform`

---

## Reproducibility & standardization

* **Pin runtime**: single‑core affinity and high priority from YAML.
* **Large inputs**: warn if <10 MB — small files inflate overheads.
* **Document**: meta captures CPU model, logical/physical cores, freq, RAM, OS, power plan/governor, lib versions, warm‑up and repeats.

This is sufficient for an IB EE to argue that speed measurements are fair and the **ratio vs. entropy** results are hardware‑independent (given fixed library versions).

---

## Interpreting metrics

* **bits/byte (bpb)** ≈ average information per original byte.

  * Lower bpb ⇒ more compressible. Ideal lower bound: `N_bytes * bpb / 8`.
  * For binary IID Bernoulli(p): `bpb = h2(p)` for ascii01, `8*h2(p)` for bitpack.
* **% over entropy**: `(achieved_bpb − H₁) / H₁` — smaller is better.
* **Order matters**: Same marginal `p` with different sequencing can compress very differently (runs vs. iid). This is captured by the sequential orders in generated data.

---

## .gitignore tips

Add these (adjust as needed):

```
# Python
__pycache__/
*.py[cod]
*.egg-info/
.eggs/

# Virtual envs
.venv/
venv/

# IDE
.vscode/

# Outputs / large data
results/
data/binary/
*.csv
*.log
```

---

## Troubleshooting

* **“No input files matched”**: ensure `make_binary_data.py` ran and `input_globs` include `**` patterns; the sweeps expand globs with `recursive=True`.
* **Import errors**: `pip install -r requirements.txt` inside the virtualenv.
* **Weird speeds**: verify High Performance power plan (Windows), ensure single‑core affinity is applied, and use ≥10 MB files.

---

## License

MIT (or your preference).
