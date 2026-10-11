#!/usr/bin/env python3
"""Evaluate a prefix-conditioned Repeat-teacher latent adapter."""
from __future__ import annotations
import argparse, json, os, sys, time
from collections import defaultdict
from pathlib import Path
from typing import Any
import torch
from modelscope import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from kv_semantic_attack.prefix_adapter_repeat import (  # noqa: E402
  PrefixLatentAdapter,
  append_dynamic_latents_batch,
  target_prefix_states_batch,
)
from kv_semantic_attack.train_universal_soft_suffix import read_records  # noqa: E402
from kv_semantic_attack.universal_soft_suffix import freeze_model  # noqa: E402
from scripts import batch_eval as batch_ops  # noqa: E402
from scripts import run_direct_reuse as base  # noqa: E402
from kv_semantic_attack.tensor_adapter_inference import infer_batch


def args() -> argparse.Namespace:
 p=argparse.ArgumentParser(description=__doc__); p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--input',type=Path,action='append',required=True); p.add_argument('--include-dataset',action='append',default=[]); p.add_argument('--model-path',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--reasoning',choices=('no','yes'),default='no'); p.add_argument('--max-new-tokens',type=int,default=4096); p.add_argument('--batch-size',type=int,default=8); p.add_argument('--max-records',type=int,default=0); p.add_argument('--directions-file',type=Path,help='JSONL rows containing task_id and target (a or b); evaluate only these directions'); p.add_argument('--inference-backend',choices=('tensor','legacy'),default='tensor'); return p.parse_args()

def parts(tok: Any,rec:dict[str,Any],side:str,reasoning:bool)->base.PromptParts:
 # reasoning selects the visible prompt, never Qwen's native thinking mode.
 return base.build_prompt_parts(tok,rec,side,enable_thinking=False,explicit_reasoning=reasoning,boxed_output=True)


def output_prefix(reasoning: bool) -> str:
  """Restore the assistant prefill that is outside generated token IDs."""
  return "" if reasoning else "The answer is: \\boxed{"


def stop_at_closing_brace(record: dict[str, Any], reasoning: bool) -> bool:
 """Only the direct-answer prefill can end at an arbitrary right brace."""
 return not reasoning and record.get('output_protocol') != 'json_answer'


def generation_result(tok: Any, rec: dict[str, Any], side: str,
                      tokens: list[int], limit: int, reasoning: bool) -> dict[str, Any]:
 return base.result_from_generation(
  tok, rec['dataset'], rec[f'gold_{side}'], tokens, len(tokens) >= limit,
  0.0, limit, enable_thinking=False, explicit_reasoning=reasoning,
  output_prefix=output_prefix(reasoning), output_protocol=rec.get('output_protocol'))


def ensure_protocol(output: Path, protocol: dict[str, Any]) -> None:
 """Fail closed instead of mixing corrected prompts with legacy samples."""
 path=output/'eval_protocol.json'
 if path.exists():
  if json.loads(path.read_text(encoding='utf-8')) != protocol:
   raise ValueError('Evaluation protocol changed; use a new output directory')
 else:
  samples=output/'samples.jsonl'
  if (samples.exists() and samples.stat().st_size) or (output/'summary.json').exists():
   raise ValueError('Legacy outputs have no verified protocol; use a new output directory')
  base.write_json_atomic(path,protocol)


def load_completed(path: Path, rows: list[tuple[dict[str, Any], str]]) -> list[dict[str, Any]]:
 """Load a valid JSONL prefix and discard a partially written final record."""
 if not path.exists(): return []
 completed=[]; valid_bytes=0
 with path.open('rb') as handle:
  while line := handle.readline():
   try: result=json.loads(line)
   except (UnicodeDecodeError,json.JSONDecodeError): break
   index=len(completed)
   if index>=len(rows): raise ValueError(f'{path}: more results than requested rows')
   record,side=rows[index]
   if result.get('task_id')!=record['task_id'] or result.get('target')!=side:
    raise ValueError(f'{path}: result {index} does not match deterministic input order')
   completed.append(result); valid_bytes=handle.tell()
 if path.stat().st_size!=valid_bytes:
  with path.open('r+b') as handle: handle.truncate(valid_bytes)
 return completed


def append_results(path: Path, results: list[dict[str, Any]]) -> None:
 """Durably checkpoint one completed evaluation batch."""
 with path.open('a',encoding='utf-8') as handle:
  handle.write(''.join(json.dumps(result,ensure_ascii=False)+'\n' for result in results))
  handle.flush(); os.fsync(handle.fileno())


def main()->int:
 a=args(); ckpt=torch.load(a.checkpoint,map_location='cpu',weights_only=False); c=ckpt['config']; rows=[]; seen=set()
 for path in a.input:
  for rec in read_records(path):
   if a.include_dataset and rec['dataset'] not in set(a.include_dataset): continue
   if rec['task_id'] in seen: raise ValueError('duplicate task_id')
   seen.add(rec['task_id']); rows.extend((rec,s) for s in ('a','b'))
 if a.directions_file:
  selected=set()
  for line in a.directions_file.read_text(encoding='utf-8').splitlines():
   if line.strip():
    row=json.loads(line)
    target=row.get('target')
    if target not in {'a','b'}: raise ValueError(f'{a.directions_file}: target must be a or b')
    selected.add((row['task_id'],target))
  rows=[(rec,side) for rec,side in rows if (rec['task_id'],side) in selected]
  if len(rows)!=len(selected): raise ValueError(f'{a.directions_file}: requested directions missing from inputs')
 if a.max_records:
  rows=rows[:2*a.max_records]
 a.output.mkdir(parents=True,exist_ok=True)
 protocol={'version':'explicit_prompt_thinking_off_v1','enable_thinking':False,
           'explicit_reasoning':a.reasoning=='yes','stop_policy':'brace_only_for_non_json_direct_prefill',
           'max_new_tokens':a.max_new_tokens,'checkpoint':str(a.checkpoint.resolve()),
           'model_path':str(a.model_path.resolve()),'inputs':[str(p.resolve()) for p in a.input],
           'include_dataset':sorted(a.include_dataset),'max_records':a.max_records,
           'directions_file':str(a.directions_file.resolve()) if a.directions_file else None}
 protocol['inference_backend']=a.inference_backend
 protocol['batch_size']=a.batch_size
 ensure_protocol(a.output,protocol)
 samples_path=a.output/'samples.jsonl'; out=load_completed(samples_path,rows); tok=AutoTokenizer.from_pretrained(a.model_path,local_files_only=True); model=AutoModelForCausalLM.from_pretrained(a.model_path,local_files_only=True,torch_dtype=torch.bfloat16,device_map='cuda:0',attn_implementation='sdpa'); freeze_model(model); base.assert_default_rope(model)
 adapter=PrefixLatentAdapter(model.config.hidden_size,c['latent_tokens'],c['adapter_bottleneck']).to(base.input_device(model)); adapter.load_state_dict(ckpt['adapter']); adapter.eval()
 correct=sum(bool(result['correct']) for result in out); parsed=sum(result['prediction'] is not None for result in out); resumed=len(out)
 inference_started=time.perf_counter()
 progress=tqdm(range(resumed,len(rows),a.batch_size),desc='adapter-eval',unit='batch',initial=resumed//a.batch_size,total=(len(rows)+a.batch_size-1)//a.batch_size)
 for start in progress:
  chunk=rows[start:start+a.batch_size]; targets=[parts(tok,rec,side,a.reasoning=='yes') for rec,side in chunk]; sources=[parts(tok,rec,'b' if side=='a' else 'a',a.reasoning=='yes') for rec,side in chunk]
  if a.inference_backend=='tensor':
   generated=infer_batch(model,adapter,tok,targets,sources,c.get('prefix_pooling','last'),
                         a.max_new_tokens,[stop_at_closing_brace(rec,a.reasoning=='yes') for rec,_ in chunk])
  else:
   with torch.no_grad():
    _,source_layers=batch_ops.prefill_ids_batch(model,[p.full_ids for p in sources]); _,prefix_layers=batch_ops.prefill_ids_batch(model,[p.prefix_ids for p in targets]); mixed=[]
    for target,source,source_cache,prefix_cache in zip(targets,sources,source_layers,prefix_layers):
     if not torch.equal(target.block_ids,source.block_ids): raise ValueError('block token mismatch')
     block=base.slice_cache(source_cache,source.block_token_start,source.block_token_end); block=base.relocate_block(model,block,source.block_token_start,target.block_token_start); mixed.append(base.splice_prefix_block(prefix_cache,block))
   states=target_prefix_states_batch(model,[p.prefix_ids for p in targets],pooling=c.get('prefix_pooling','last')); latents=torch.stack([adapter(state) for state in states]); logits,caches,lengths=append_dynamic_latents_batch(model,mixed,[p.suffix_ids for p in targets],latents); padded=batch_ops.pad_cache(list(zip(caches,lengths)),max(lengths)); generated=batch_ops.decode_batch(model,tok,logits,padded,lengths,a.max_new_tokens,stop_at_closing_brace=[stop_at_closing_brace(rec,a.reasoning=='yes') for rec,_ in chunk])
  batch_results=[]
  for (rec,side),tokens in zip(chunk,generated):
    result=generation_result(tok,rec,side,tokens,a.max_new_tokens,a.reasoning=='yes'); result.update({'task_id':rec['task_id'],'dataset':rec['dataset'],'target':side,'source':'b' if side=='a' else 'a'}); batch_results.append(result); correct+=int(bool(result['correct'])); parsed+=int(result['prediction'] is not None)
  append_results(samples_path,batch_results); out.extend(batch_results)
  live={'directions':len(out),'accuracy':correct/len(out),'parse_rate':parsed/len(out)}; progress.set_postfix(acc=f'{live["accuracy"]:.4f}',parse=f'{live["parse_rate"]:.4f}'); print(json.dumps(live),flush=True)
 by=defaultdict(list)
 for r in out: by[r['dataset']].append(r)
 one=lambda rs:{'directions':len(rs),'correct':sum(bool(r['correct']) for r in rs),'accuracy':sum(bool(r['correct']) for r in rs)/len(rs),'parse_rate':sum(r['prediction'] is not None for r in rs)/len(rs)}
 summary={'complete':True,'checkpoint':str(a.checkpoint.resolve()),'batch_size':a.batch_size,'reasoning':a.reasoning,'enable_thinking':False,'evaluation_protocol':protocol,'max_new_tokens':a.max_new_tokens,'overall':one(out),'by_dataset':{k:one(v) for k,v in sorted(by.items())}}
 summary['inference_backend']=a.inference_backend
 summary['inference_seconds']=time.perf_counter()-inference_started
 summary['new_directions']=len(out)-resumed
 summary['directions_per_second']=(len(out)-resumed)/summary['inference_seconds']
 base.write_json_atomic(a.output/'summary.json',summary); print(json.dumps(summary,ensure_ascii=False,indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
