import json
from itertools import islice

# Path to the JSONL file (relative to this notebook's directory)
file_path = 'qa_50000.jsonl'
num_samples = 10

samples = []
with open(file_path, 'r', encoding='utf-8') as f:
    for line in islice(f, num_samples):
        line = line.strip()
        if not line:
            continue
        try:
            samples.append(json.loads(line))
        except json.JSONDecodeError as e:
            samples.append({"error": f"JSON decode error: {e}", "raw": line})

for idx, sample in enumerate(samples, start=1):
    print(f"[{idx}]")
    print(json.dumps(sample, ensure_ascii=False, indent=2))
    print('-' * 40)
