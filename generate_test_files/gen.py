# generate_test_files/gen.py
from pathlib import Path
import random, math, bisect, string, json
from collections import Counter

# --- Choose where to write files (pick ONE style) ---
# 1) Raw string (good for Windows paths with backslashes and spaces/diacritics)
BASE_DIR = Path(r"C:\Users\Marek Ištok\Documents\Visual studio code\Brotli\data")

ALPHABET = list(string.ascii_lowercase)  # 26 letters
N = 1_000_000       # characters per file
WRAP = 80           # newline every 80 chars
SEED = 42           # set a seed for reproducibility

def write_wrapped(path: Path, chars):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for i in range(0, len(chars), WRAP):
            f.write(''.join(chars[i:i+WRAP]) + '\n')

def make_uniform_exact(n=N):
    # Equal counts across 26 letters (as equal as possible), then shuffle
    q, r = divmod(n, 26)
    counts = [q + (1 if i < r else 0) for i in range(26)]
    out = []
    for i, c in enumerate(counts):
        out.extend(ALPHABET[i] * c)
    random.shuffle(out)
    return out

def sample_from_weights(weights, n):
    total = sum(weights)
    cdf, acc = [], 0.0
    for w in weights:
        acc += w / total
        cdf.append(acc)
    out = []
    for _ in range(n):
        r = random.random()
        j = bisect.bisect_left(cdf, r)
        out.append(ALPHABET[j])
    return out

def make_gaussian(n=N, mu=12.5, sigma=5.0):
    weights = [math.exp(-0.5 * ((i - mu)/sigma)**2) for i in range(26)]
    return sample_from_weights(weights, n)

def make_linear(n=N):
    weights = [i+1 for i in range(26)]  # p(i) ∝ (i+1)
    return sample_from_weights(weights, n)

def summarize(chars):
    cnt = Counter(chars)
    total = sum(cnt.values())
    return {k: round(v / total, 4) for k, v in sorted(cnt.items())}

def main():
    random.seed(SEED)

    uniform_chars  = make_uniform_exact(N)
    gaussian_chars = make_gaussian(N)
    linear_chars   = make_linear(N)

    uniform_path  = BASE_DIR / "letters_uniform.txt"
    gaussian_path = BASE_DIR / "letters_gaussian.txt"
    linear_path   = BASE_DIR / "letters_linear.txt"

    write_wrapped(uniform_path, uniform_chars)
    write_wrapped(gaussian_path, gaussian_chars)
    write_wrapped(linear_path, linear_chars)

    report = {
        "created": [str(uniform_path), str(gaussian_path), str(linear_path)],
        "freq_samples": {
            "uniform":  summarize(uniform_chars),
            "gaussian": summarize(gaussian_chars),
            "linear":   summarize(linear_chars),
        }
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
