from __future__ import annotations
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from abpr.model import find_task2_split_files, load_query_csv, track2_metrics
from abpr.runtime import ABPRRuntime


def main():
    p=argparse.ArgumentParser(description='Score the exact exported runtime on labeled public Track-2 validation.')
    p.add_argument('--repo-root',default='/content/UPAR-Challenge-2027'); p.add_argument('--model',required=True); p.add_argument('--output',default='')
    a=p.parse_args(); root=Path(a.repo_root)
    gallery_csv,query_csv=find_task2_split_files(root,'val'); gdf=pd.read_csv(gallery_csv); _,queries=load_query_csv(query_csv)
    from abpr.core import ATTRIBUTE_NAMES, detect_image_column, ImageResolver
    image_col=detect_image_column(gdf); resolver=ImageResolver(root/'data',root)
    samples=[]
    for x in tqdm(gdf[image_col].astype(str).tolist(),desc='Resolve public gallery',dynamic_ncols=True):
        samples.append({'image_path':str(resolver.resolve(x))})
    labels=gdf[ATTRIBUTE_NAMES].to_numpy(np.int16)
    runtime=ABPRRuntime(a.model)
    probs,emb=runtime.encode_gallery(samples)
    d=runtime.distance(queries,probs,emb)
    result={'model':str(a.model),'gallery':str(gallery_csv),'queries':str(query_csv),'gallery_size':len(samples),'query_size':len(queries),'metrics':track2_metrics(d,queries,labels)}
    print(json.dumps(result,indent=2))
    if a.output:
        Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(result,indent=2),encoding='utf-8')

if __name__=='__main__': main()
