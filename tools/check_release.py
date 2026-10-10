#!/usr/bin/env python3
"""Offline checks of release source, config CLI compatibility and local links."""

import sys
from pathlib import Path

# Support running this entry point directly from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import ast
import json
import re

ROOT = Path(__file__).resolve().parents[1]
SKIP = {'.git', '.venv', '.deps', '.cache', '__pycache__', 'model_weight', 'checkpoints', 'output', 'data', 'build', 'dist'}


def source_files():
    for path in ROOT.rglob('*'):
        rel = path.relative_to(ROOT)
        if path.is_file() and rel.parts[0] not in SKIP and '__pycache__' not in rel.parts and not any(x.endswith('.egg-info') for x in rel.parts):
            yield path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    errors = []
    files = list(source_files())
    py_count = 0
    for path in files:
        rel = str(path.relative_to(ROOT))
        if path.is_symlink():
            errors.append(f'{rel}: symlink must be reviewed')
        if path.stat().st_size > 10 * 1024 * 1024:
            errors.append(f'{rel}: unexpectedly large release file')
        if path.suffix not in {'.py', '.json', '.md', '.toml', '.txt', '.sh', '.yaml', '.csv'}:
            continue
        text = path.read_text()
        if path.suffix == '.py':
            py_count += 1
            try:
                compile(text, rel, 'exec')
            except SyntaxError as exc:
                errors.append(f'{rel}:{exc.lineno}: syntax error')
        if path.suffix == '.json':
            try:
                json.loads(text)
            except ValueError:
                errors.append(f'{rel}: malformed JSON')
        # Report locations only; never print credential values.
        for num, line in enumerate(text.splitlines(), 1):
            if re.search(r'(?:AKIA|ASIA)[A-Z0-9]{16}|gh[pousr]_[A-Za-z0-9]{30,}|hf_[A-Za-z0-9]{25,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY', line):
                errors.append(f'{rel}:{num}: possible secret')
            for uri in re.findall(r'(?:s3|cos)://([^\s\x22\x27]+)', line):
                if '@' in uri and not any(x in uri for x in ['<', 'SECRET', 'Secret', 'ACCESS_KEY']):
                    errors.append(f'{rel}:{num}: credential-bearing storage URI')
        if path.suffix == '.md':
            for target in re.findall(r'(?<!!)\[[^\]]*\]\(([^ )]+)', text):
                if '://' in target or target.startswith('#'):
                    continue
                target = target.split('#')[0]
                if not (path.parent / target).exists():
                    errors.append(f'{rel}: missing link target {target}')
    # Keep the public entry points small and prevent imports of executable scripts.
    top_level = {p.name for p in (ROOT / 'scripts').glob('*.py')}
    if top_level != {'train.py', 'inference.py'}:
        errors.append('scripts/: expected only train.py and inference.py at the top level')
    for path in [*(ROOT / 'linmorph').glob('*.py'), *(ROOT / 'scripts').rglob('*.py')]:
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue  # Already reported by the syntax check above.
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ''
                if module == 'linmorph':
                    names = ['linmorph.' + alias.name for alias in node.names]
                elif module.startswith('linmorph.'):
                    names = [module]
                elif node.level == 1 and path.parent == ROOT / 'linmorph':
                    names = ['linmorph.' + module]
                else:
                    names = []
                for name in names:
                    target = ROOT.joinpath(*name.split('.'))
                    if not target.with_suffix('.py').is_file() and not (target / '__init__.py').is_file():
                        errors.append(f'{path.relative_to(ROOT)}:{node.lineno}: missing internal module {name}')
                if module.startswith('scripts.'):
                    errors.append(f'{path.relative_to(ROOT)}:{node.lineno}: move shared code into linmorph/')
    manifest_path = ROOT / 'docs/source_manifest.json'
    for row in json.loads(manifest_path.read_text()):
        for destination in [row['destination'], *row.get('additional_destinations', [])]:
            if not (ROOT / destination).is_file():
                errors.append(f'source_manifest.json: missing destination {destination}')
    training = ROOT / 'examples/wanvideo/model_training'
    for config in (ROOT / 'configs/training').glob('*.json'):
        cfg = json.loads(config.read_text())
        paths = [training / cfg['entrypoint'], training / 'train.py', ROOT / 'diffsynth/diffusion/parsers.py']
        options = set()
        for path in paths:
            tree = ast.parse(path.read_text())
            for call in ast.walk(tree):
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == 'add_argument':
                    options.update(a.value[2:] for a in call.args if isinstance(a, ast.Constant) and isinstance(a.value, str) and a.value.startswith('--'))
        unknown = set(cfg['arguments']) - options
        if unknown:
            errors.append(f'{config.name}: unknown training options {sorted(unknown)}')
    for config in (ROOT / 'configs/inference').glob('*.json'):
        cfg = json.loads(config.read_text())
        blocks = list(map(int, cfg['model']['gdn_block_indices'].split(',')))
        if len(set(blocks)) != len(blocks) or any(b < 0 or b >= 30 for b in blocks):
            errors.append(f'{config.name}: invalid block list')
    print(json.dumps({'files_checked': len(files), 'python_files_compiled': py_count, 'errors': errors}, indent=2))
    return bool(errors)


if __name__ == '__main__':
    sys.exit(main())
