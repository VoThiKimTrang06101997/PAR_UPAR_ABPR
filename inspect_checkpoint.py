from pathlib import Path
import argparse, json, torch

p=argparse.ArgumentParser(); p.add_argument('checkpoint'); a=p.parse_args()
ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
summary={
    'path':str(Path(a.checkpoint)),
    'format':ck.get('format'),
    'epoch':ck.get('epoch'),
    'global_step':ck.get('global_step'),
    'backbone':ck.get('backbone'),
    'embed_dim':ck.get('embed_dim'),
    'image_height':ck.get('image_height'),
    'image_width':ck.get('image_width'),
    'num_models':len(ck.get('models',[])) if ck.get('models') is not None else 1,
    'validation':ck.get('validation'),
    'distance_kind':ck.get('distance_kind'),
    'prototype_mix':ck.get('prototype_mix'),
    'tta_flip':ck.get('tta_flip'),
    'source_checkpoints':ck.get('source_checkpoints'),
}
print(json.dumps(summary,indent=2,default=str))
