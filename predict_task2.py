from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from abpr.model import load_query_csv
from abpr.runtime import ABPRRuntime
from abpr.core import detect_image_column, ImageResolver


def main():
    p=argparse.ArgumentParser(); p.add_argument('--repo-root',required=True); p.add_argument('--model',required=True); p.add_argument('--gallery-csv',required=True); p.add_argument('--query-csv',required=True); p.add_argument('--output-dir',required=True); p.add_argument('--topk',type=int,default=100)
    a=p.parse_args(); root=Path(a.repo_root); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    gdf=pd.read_csv(a.gallery_csv); _,queries=load_query_csv(a.query_csv); image_col=detect_image_column(gdf); resolver=ImageResolver(root/'data',root)
    samples=[]
    for x in tqdm(gdf[image_col].astype(str).tolist(),desc='Resolve gallery',unit='image',dynamic_ncols=True):
        samples.append({'image_path':str(resolver.resolve(x))})
    runtime=ABPRRuntime(a.model); probs,emb=runtime.encode_gallery(samples); d=runtime.distance(queries,probs,emb).numpy()
    topk=min(int(a.topk),d.shape[1]); ranking=np.argsort(d,axis=1,kind='stable')[:,:topk]
    np.save(out/'rankings_topk.npy',ranking.astype(np.int32)); np.save(out/'distances.npy',d.astype(np.float32))
    print('queries=',len(queries),'gallery=',len(samples),'topk=',topk,'output=',out)

if __name__=='__main__': main()
