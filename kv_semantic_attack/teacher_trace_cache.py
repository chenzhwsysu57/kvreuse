"""Validated token-only Full teacher cache; logits are recomputed by forward."""
import hashlib
import json
from pathlib import Path


def cache_protocol(records, model_path, teacher_tokens):
    path = Path(model_path).resolve()
    model_files = []
    for file in sorted(path.iterdir()):
        if file.is_file() and (file.suffix in {'.json', '.safetensors', '.bin', '.model'}):
            stat = file.stat()
            model_files.append([file.name, stat.st_size, stat.st_mtime_ns])
    return {'version': 'full_reasoning_token_cache_v1', 'enable_thinking': False,
            'explicit_reasoning': True, 'teacher_tokens': teacher_tokens,
            'model_path': str(path), 'model_files': model_files,
            'records_sha256': hashlib.sha256(json.dumps(records, sort_keys=True,
                                                       ensure_ascii=False).encode()).hexdigest()}


def load_cache(path, protocol, expected):
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    if payload['protocol'] != protocol:
        raise ValueError('Full teacher cache protocol/input/model changed')
    result = {}
    for row in payload['traces']:
        key = (row['task_id'], row['target'])
        tokens = row['token_ids']
        if key in result or key not in expected:
            raise ValueError('duplicate or unknown Full teacher cache key')
        if not tokens or len(tokens) > protocol['teacher_tokens'] or any(
                not isinstance(t, int) or isinstance(t, bool) or t < 0 for t in tokens):
            raise ValueError('invalid Full teacher trace token IDs')
        result[key] = tokens
    if payload.get('complete') and set(result) != expected:
        raise ValueError('complete Full teacher cache has missing directions')
    return result


def save_cache(path, protocol, traces, expected):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {'protocol': protocol, 'complete': set(traces) == expected,
               'traces': [{'task_id': task, 'target': side, 'token_ids': tokens}
                          for (task, side), tokens in sorted(traces.items())]}
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(payload, ensure_ascii=False)+'\n')
    tmp.replace(path)