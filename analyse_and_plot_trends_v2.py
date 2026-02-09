#!/usr/bin/env python3
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# --- Scikit-Learn Imports ---
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

# --- Configuration ---
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("plots/trends_sklearn")
STATS_FILE = Path("analysis_summary_sklearn.txt")

# Define File Sources (Matches your specific folder structure)
SOURCES = {
    "Binary_Random": [
        RESULTS_DIR / "random/results_brotli.csv",
        RESULTS_DIR / "random/results_lz4.csv"
    ],
    "Binary_Sequential": [
        RESULTS_DIR / "seq/results_brotli.csv",
        RESULTS_DIR / "seq/results_lz4.csv"
    ],
    "Synthetic_Kary": [
        RESULTS_DIR / "kary/results_brotli.csv",
        RESULTS_DIR / "kary/results_lz4.csv"
    ]
}

def identify_sub_experiment(group_name, dataset_path):
    """Refines the experiment type based on the filename/path."""
    s = dataset_path.lower().replace("\\", "/")
    
    if group_name == "Binary_Sequential":
        if "alternating" in s: return "Binary_Seq_Alternating"
        if "blocks" in s: return "Binary_Seq_Blocks"
        return "Binary_Seq_Other"
    
    if group_name == "Binary_Random":
        return "Binary_Random_IID"
        
    if group_name == "Synthetic_Kary":
        if "iid_peaked" in s: return "Kary_IID_Peaked"
        if "zipf" in s: return "Kary_Zipf"
        if "markov" in s: return "Kary_Markov"
        return "Kary_Other"
        
    return "Other"

def get_quality(row):
    """Robustly extracts quality/level from params JSON."""
    try:
        params = json.loads(row.get('params', '{}'))
        if 'quality' in params: return int(params['quality'])
        if 'level' in params: return int(params['level'])
        if 'compression_level' in params: return int(params['compression_level'])
    except:
        pass
    return None

def load_data():
    """
    Returns nested dict: 
    data[SubExperiment][Alg][Quality] = {'H': [], 'gap': []}
    """
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {'H': [], 'gap': []})))
    
    for group_name, files in SOURCES.items():
        for p in files:
            if not p.exists():
                print(f"[Warn] File not found: {p}")
                continue
                
            with open(p, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        alg = row.get('alg', '').lower()
                        bpb = float(row['bpb'])
                        H = float(row['entropy_bpb'])
                        dataset = row['dataset']
                        
                        # Filter invalid data (0 entropy or compression error)
                        if H <= 0 or bpb <= 0: continue
                        
                        q = get_quality(row)
                        if q is None: continue

                        sub_exp = identify_sub_experiment(group_name, dataset)
                        if "Other" in sub_exp: continue 

                        gap = bpb - H
                        
                        data[sub_exp][alg][q]['H'].append(H)
                        data[sub_exp][alg][q]['gap'].append(gap)
                        
                    except (ValueError, KeyError):
                        continue
    return data

def fit_and_plot(sub_exp, alg, q, H_data, gap_data):
    """Fits Linear and Poly models using SKLEARN, saves plot, returns stats string."""
    
    if len(H_data) < 4: return None 

    # Sort data for clean plotting
    sorted_indices = np.argsort(H_data)
    X = np.array(H_data)[sorted_indices].reshape(-1, 1) # Sklearn needs 2D array (n_samples, n_features)
    y = np.array(gap_data)[sorted_indices]

    # --- 1. Linear Regression (Degree 1) ---
    lin_model = LinearRegression()
    lin_model.fit(X, y)
    y_pred_lin = lin_model.predict(X)
    r2_lin = r2_score(y, y_pred_lin)
    
    # Extract coefficients for label (y = mx + c)
    m = lin_model.coef_[0]
    c = lin_model.intercept_

    # --- 2. Polynomial Regression (Degree 2) ---
    # Pipeline: create features [x, x^2] -> Linear Regression
    poly_model = make_pipeline(PolynomialFeatures(2), LinearRegression())
    poly_model.fit(X, y)
    y_pred_poly = poly_model.predict(X)
    r2_poly = r2_score(y, y_pred_poly)

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    
    # Scatter Data
    ax.scatter(X, y, color='black', alpha=0.6, s=25, label='Measured Data')
    
    # Generate smooth X range for plotting trend lines
    X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    
    # Plot Linear Fit
    ax.plot(X_plot, lin_model.predict(X_plot), "r--", linewidth=1.5, 
            label=f'Linear ($R^2={r2_lin:.3f}$)\n$y={m:.3f}x + {c:.3f}$')
    
    # Plot Poly Fit
    ax.plot(X_plot, poly_model.predict(X_plot), "b-", alpha=0.7, linewidth=1.5,
            label=f'Poly Deg2 ($R^2={r2_poly:.3f}$)')

    # Formatting
    ax.set_title(f"{sub_exp} | {alg.upper()} | Q={q}")
    ax.set_xlabel("Zero-Order Entropy ($H_0$) [bits/byte]")
    ax.set_ylabel("Redundancy Gap ($BPB - H_0$)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    
    # Save
    safe_name = f"{sub_exp}_{alg}_Q{q}.png"
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / safe_name)
    plt.close(fig)

    # Return Stats String for the summary file
    best_fit = "Linear" if r2_lin >= r2_poly else "Poly"
    
    # Note: If difference is negligible (< 0.02), prefer Linear (Occam's razor)
    if abs(r2_poly - r2_lin) < 0.02: 
        best_fit = "Linear (Tie)"

    return (f"{sub_exp:<22} | {alg.upper():<6} | Q={q:<2} || "
            f"Lin R2: {r2_lin:.4f} | Poly R2: {r2_poly:.4f} | Best: {best_fit}")

def main():
    print("Starting analysis with Scikit-Learn...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()
    
    with open(STATS_FILE, "w", encoding="utf-8") as f:
        f.write("=================================================================================\n")
        f.write("   ANALYSIS SUMMARY (SKLEARN): Linear vs Polynomial Trends per Quality Level     \n")
        f.write("=================================================================================\n")
        f.write(f"{'Experiment':<22} | {'Alg':<6} | {'Q':<4} || {'Stats':<40}\n")
        f.write("-" * 90 + "\n")
        
        count = 0
        # Sort keys for consistent output order
        for sub_exp in sorted(data.keys()):
            for alg in sorted(data[sub_exp].keys()):
                for q in sorted(data[sub_exp][alg].keys()):
                    
                    H = data[sub_exp][alg][q]['H']
                    gap = data[sub_exp][alg][q]['gap']
                    
                    stats_line = fit_and_plot(sub_exp, alg, q, H, gap)
                    
                    if stats_line:
                        print(stats_line)
                        f.write(stats_line + "\n")
                        count += 1

    print(f"\nDone! Generated {count} graphs.")
    print(f"Graphs saved to: {OUTPUT_DIR.absolute()}")
    print(f"Stats saved to:  {STATS_FILE.absolute()}")

if __name__ == "__main__":
    main()