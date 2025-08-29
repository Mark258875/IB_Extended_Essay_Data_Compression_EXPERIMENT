# Brotli EE Mini-Bench (Python)

Bench Brotli (and baselines) on your files or synthetic data. Reports:
- ratio, bits/byte (bpb)
- zero-order entropy (bits/byte)
- % over entropy
- compression/decompression throughput (MB/s)

### Install
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

NOTES:

- nezalezi to len od mnozstva ale aj od kombinaciiiii !!! 