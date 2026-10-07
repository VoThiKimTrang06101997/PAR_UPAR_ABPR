from __future__ import annotations

from pathlib import Path
import argparse, ast, json, shutil, zipfile
import numpy as np
import torch


def locate_metadata(official_root: Path):
    preferred=official_root/'examples'/'task2'/'sample_code_submission'/'metadata.yaml'
    if preferred.exists(): return preferred
    candidates=[p for p in official_root.rglob('metadata.yaml') if 'task2' in p.as_posix().lower()]
    return candidates[0] if candidates else None


def main():
    p=argparse.ArgumentParser(); p.add_argument('--official-root',default='/content/UPAR-Challenge-2027'); p.add_argument('--checkpoint',required=True)
    p.add_argument('--result-dir',default='/content/drive/MyDrive/PedestrianAttributeRecognition/ABPR_Results'); a=p.parse_args()
    official=Path(a.official_root); ck_path=Path(a.checkpoint); out=Path(a.result_dir); out.mkdir(parents=True,exist_ok=True)
    ck=torch.load(ck_path,map_location='cpu',weights_only=False)
    if not ck.get('models'): raise RuntimeError('Expected calibrated checkpoint containing one or more ensemble models.')
    required40=('attribute_weights','calibration_scale','calibration_bias','priors','attribute_names')
    for key in required40:
        if len(ck.get(key,[]))!=40: raise RuntimeError(f'{key} must contain 40 values')
    package={k:ck[k] for k in [
        'format','models','image_height','image_width','attribute_names','attribute_weights','calibration_scale','calibration_bias',
        'distance_kind','embedding_mix','prototype_mix','positive_rarity_power','negative_weight','tta_flip','priors','validation','retrieval_calibration','source_checkpoints'
    ] if k in ck}
    package['format']='upar2027-track2-abpr-prototype-runtime'

    work=Path('/content/UPAR2027_Track2_ABPR_runtime') if Path('/content').exists() else out/'runtime_tmp'
    if work.exists(): shutil.rmtree(work)
    (work/'assets').mkdir(parents=True); torch.save(package,work/'assets'/'model.pt')
    here=Path(__file__).resolve().parent; shutil.copy2(here/'abpr'/'runtime.py',work/'abpr_runtime.py'); shutil.copy2(here/'submission_run.py',work/'run.py')
    metadata=locate_metadata(official)
    if metadata is None: raise FileNotFoundError('Could not locate official Task-2 metadata.yaml')
    shutil.copy2(metadata,work/'metadata.yaml')
    run_text=(work/'run.py').read_text(encoding='utf-8'); funcs=[n.name for n in ast.parse(run_text).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]
    if 'rank_gallery' not in funcs or 'ABPRRuntime' not in run_text or 'model.pt' not in run_text: raise RuntimeError('run.py is not model-backed')
    if 'attribute_prior.json' in run_text or '_PRIOR' in run_text: raise RuntimeError('Prior-only organizer baseline leaked into runtime')
    info={'checkpoint':str(ck_path),'num_models':len(package['models']),'distance_kind':package.get('distance_kind'),'embedding_mix':package.get('embedding_mix'),'prototype_mix':package.get('prototype_mix'),'positive_rarity_power':package.get('positive_rarity_power'),'negative_weight':package.get('negative_weight'),'tta_flip':package.get('tta_flip'),'validation':package.get('validation'),'run_functions':funcs}
    (work/'RUNTIME_INFO.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    final=out/'UPAR2027_Track2_ABPR_Runtime.zip'; final.unlink(missing_ok=True)
    with zipfile.ZipFile(final,'w',zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as zf:
        for f in sorted(work.rglob('*')):
            if f.is_file(): zf.write(f,f.relative_to(work).as_posix())
    print('Runtime bundle:',final); print(json.dumps(info,indent=2))

if __name__=='__main__': main()
