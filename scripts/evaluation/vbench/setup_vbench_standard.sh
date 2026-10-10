#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python}"
"$PYTHON" -m pip install --upgrade --no-deps --target .deps/wan_unipc 'diffusers==0.31.0'
"$PYTHON" - <<'PY'
import hashlib
import json
from pathlib import Path
import urllib.request
root = Path('.cache/vbench/official_protocol')
root.mkdir(parents=True, exist_ok=True)
urls = {
    'fm_solvers_unipc.py': 'https://raw.githubusercontent.com/Wan-Video/Wan2.1/main/wan/utils/fm_solvers_unipc.py',
    'all_dimension_aug_wanx_seed42.txt': 'https://raw.githubusercontent.com/Vchitect/VBench/master/prompts/augmented_prompts/Wan2.1-T2V-1.3B/all_dimension_aug_wanx_seed42.txt',
}
expected = {
    'fm_solvers_unipc.py': '0dec8c7ed17f6f2049275c6848113314da6ccec1c8db5bdc89df43c05c6038d9',
    'all_dimension_aug_wanx_seed42.txt': '54fb940aea8f1ede1e8728f62d3e67efacdd7b341fb8a6c91308bcba874d5d5d',
}
record = {}
for name, url in urls.items():
    path = root / name
    if not path.exists():
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != expected[name]:
            raise RuntimeError(f'Upstream checksum changed: {name}; review the protocol before updating hashes')
        path.write_bytes(data)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected[name]:
        raise RuntimeError(f'Cached resource checksum mismatch: {name}')
    record[name] = {'url': url, 'sha256': digest}
(root / 'resources.json').write_text(json.dumps(record, indent=2) + '\n')
print('Sampler resources ready; install VBench separately for evaluation.')
PY
