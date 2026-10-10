"""Repair absolute shard paths after moving a QWorld metadata cache."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil


def repair_qworld_cache_paths(cache_dir):
    """Validate all shard references and atomically rebase paths to this cache.

    Only the shard-path field in QWorld index manifests changes. Bucket order,
    row order, sample counts, and the actual annotation shards stay intact.
    """
    root = Path(cache_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".qworld_cache_paths.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        manifests = sorted(set(root.glob("*_merged_index.json")) | set(root.glob("*_index_list.json")))
        pending = []
        references = 0
        repaired = 0
        for manifest in manifests:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError(f"QWorld index must be a bucket dictionary: {manifest}")
            changed = 0
            for bucket, rows in data.items():
                if not isinstance(rows, list):
                    raise ValueError(f"Invalid QWorld bucket: {manifest}: {bucket}")
                for row in rows:
                    if not isinstance(row, list) or len(row) != 4 or not isinstance(row[1], str):
                        raise ValueError(f"Invalid QWorld shard record: {manifest}: {bucket}")
                    references += 1
                    original = Path(row[1])
                    # Prefer the local shard without stat-ing the old mount: retired
                    # NFS mounts can raise ESTALE or block even for an existence check.
                    # Moved shards retain their source/rank directory and filename.
                    candidate = root / original.parent.name / original.name
                    local_exists = candidate.resolve().is_relative_to(root) and candidate.is_file()
                    if not local_exists:
                        try:
                            original_exists = original.is_absolute() and original.is_file()
                        except OSError:
                            original_exists = False
                        if original_exists:
                            continue
                        raise FileNotFoundError(
                            f"QWorld cached shard is missing: {row[1]}; "
                            f"also not found at {candidate}. Restore the shard or rebuild "
                            "the cache before starting GPU training."
                        )
                    if row[1] != str(candidate):
                        row[1] = str(candidate)
                        changed += 1
            if changed:
                pending.append((manifest, data))
                repaired += changed
        # Validate the entire cache before changing any manifest. Keep the original
        # files for inspection and use atomic replacement for concurrent readers.
        for manifest, data in pending:
            backup = manifest.with_name(manifest.name + ".pre_path_repair.bak")
            if not backup.exists():
                shutil.copy2(manifest, backup)
            temporary = manifest.with_name(f".{manifest.name}.{os.getpid()}.tmp")
            try:
                temporary.write_text(json.dumps(data, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")
                os.chmod(temporary, manifest.stat().st_mode & 0o777)
                os.replace(temporary, manifest)
            finally:
                temporary.unlink(missing_ok=True)
        result = {"manifests": len(manifests), "references": references, "repaired_paths": repaired}
        if repaired:
            print(f"QWorld cache paths repaired: {result}", flush=True)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    args = parser.parse_args()
    print(f"QWorld cache path check passed: {repair_qworld_cache_paths(args.cache_dir)}")
