"""End-to-end package test using a fake tiny runtime (not performance claim)."""
import csv, json, os, subprocess, sys, zipfile
from pathlib import Path

def test_builder_executable_zip(tmp_path):
    source=tmp_path/'source';source.mkdir()
    (source/'abpr').mkdir()
    (source/'abpr/__init__.py').write_text('')
    (source/'abpr/core.py').write_text('''from pathlib import Path
ATTRIBUTE_NAMES=[f"Attr-{i}" for i in range(40)]
def detect_image_column(df): return "# image"
class ImageResolver:
 def __init__(self,*args,**kwargs): self.root=Path(args[0])
 def resolve(self,value): return self.root/value
''')
    (source/'submission_run.py').write_bytes((Path(__file__).resolve().parents[1]/'submission_run.py').read_bytes())
    official=tmp_path/'official'; meta=official/'examples/task2/sample_code_submission';meta.mkdir(parents=True)
    (meta/'metadata.yaml').write_text('name: abpr')
    val=official/'data/annotations/task2/val';val.mkdir(parents=True)
    (official/'data/images').mkdir(parents=True)
    import numpy as np
    from PIL import Image
    with (val/'gt.csv').open('w') as fh:
        writer=csv.writer(fh);writer.writerow(['# image']+[f'Attr-{i}' for i in range(40)])
        for i in range(8):
            path=f'images/{i}.png'
            Image.fromarray(np.full((32,16,3),i*30,dtype=np.uint8)).save(official/'data'/path)
            writer.writerow([path]+[int((i+j)%2==0) for j in range(40)])
    runtime=tmp_path/'runtime.zip'
    fake_runtime='''from pathlib import Path
import numpy as np
class ABPRRuntime:
 def __init__(self, ckpt):
  assert Path(ckpt).is_file()
  self.attribute_names=[f"Attr-{i}" for i in range(40)]
  self.device='cpu'
 def encode_gallery(self, gallery):
  from PIL import Image
  p=np.stack([np.linspace(.1,.8,40,dtype=np.float32)+np.asarray(Image.open(g['image_path']),dtype=np.float32).mean()/2550 for g in gallery])
  return p,np.zeros((len(gallery),8),dtype=np.float32)
 def distance(self,q,p,e): return np.asarray(-(q@p.T),np.float32)
'''
    with zipfile.ZipFile(runtime,'w') as z:
        z.writestr('abpr_runtime.py',fake_runtime)
        z.writestr('assets/model.pt',b'trained-mock')
        z.writestr('requirements.txt','numpy\nPillow\n')
    out=tmp_path/'output'
    command=[sys.executable,str(Path(__file__).resolve().parents[1]/'build_codabench_submission.py'),
      '--runtime-zip',str(runtime),'--result-dir',str(out),'--name','Submission_ABPR',
      '--official-root',str(official),'--source-root',str(source), '--smoke-timeout','30']
    r=subprocess.run(command,capture_output=True,text=True)
    assert r.returncode==0,r.stdout+'\n'+r.stderr
    assert (out/'Submission_ABPR.zip').exists()
    manifest=json.loads((out/'Submission_ABPR_manifest.json').read_text())
    assert manifest['trained_model_inference_smoke_test']['status']=='passed'
