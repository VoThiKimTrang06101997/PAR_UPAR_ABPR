from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import (
    efficientnet_b0, convnext_tiny, convnext_small, convnext_base, efficientnet_v2_s,
)

ATTRIBUTE_NAMES = [
    'Age-Young','Age-Adult','Age-Old','Gender-Female',
    'Hair-Length-Short','Hair-Length-Long','Hair-Length-Bald','UpperBody-Length-Short',
    'UpperBody-Color-Black','UpperBody-Color-Blue','UpperBody-Color-Brown','UpperBody-Color-Green',
    'UpperBody-Color-Grey','UpperBody-Color-Orange','UpperBody-Color-Pink','UpperBody-Color-Purple',
    'UpperBody-Color-Red','UpperBody-Color-White','UpperBody-Color-Yellow','UpperBody-Color-Other',
    'LowerBody-Length-Short','LowerBody-Color-Black','LowerBody-Color-Blue','LowerBody-Color-Brown',
    'LowerBody-Color-Green','LowerBody-Color-Grey','LowerBody-Color-Orange','LowerBody-Color-Pink',
    'LowerBody-Color-Purple','LowerBody-Color-Red','LowerBody-Color-White','LowerBody-Color-Yellow',
    'LowerBody-Color-Other','LowerBody-Type-Trousers&Shorts','LowerBody-Type-Skirt&Dress',
    'Accessory-Backpack','Accessory-Bag','Accessory-Glasses-Normal','Accessory-Glasses-Sun','Accessory-Hat'
]
NUM_ATTRIBUTES = len(ATTRIBUTE_NAMES)


class QueryEncoder(nn.Module):
    def __init__(self, embed_dim=320, negative_token_scale=.35):
        super().__init__(); self.negative_token_scale=float(negative_token_scale)
        self.pos_basis=nn.Parameter(torch.zeros(NUM_ATTRIBUTES,embed_dim)); self.neg_basis=nn.Parameter(torch.zeros(NUM_ATTRIBUTES,embed_dim))
        self.count_embed=nn.Sequential(nn.Linear(2,embed_dim),nn.GELU(),nn.Linear(embed_dim,embed_dim))
        self.mlp=nn.Sequential(nn.LayerNorm(embed_dim),nn.Linear(embed_dim,embed_dim*2),nn.GELU(),nn.Dropout(.10),nn.Linear(embed_dim*2,embed_dim))
    def forward(self,q):
        q=q.float(); valid=q>=0; pos=(q>.5)&valid; neg=(q<=.5)&valid
        tok=pos.unsqueeze(-1)*self.pos_basis.unsqueeze(0)+self.negative_token_scale*neg.unsqueeze(-1)*self.neg_basis.unsqueeze(0)
        den=(pos.float()+self.negative_token_scale*neg.float()).sum(1,keepdim=True).clamp_min(1.)
        z=tok.sum(1)/den.sqrt(); counts=torch.stack([pos.float().sum(1)/NUM_ATTRIBUTES,neg.float().sum(1)/NUM_ATTRIBUTES],dim=1)
        z=z+self.count_embed(counts)+self.mlp(z)
        return F.normalize(z,dim=-1)


class SpatialAttributeHead(nn.Module):
    def __init__(self,in_dim,value_dim=320,num_attributes=NUM_ATTRIBUTES):
        super().__init__(); self.attn=nn.Conv2d(in_dim,num_attributes,1,bias=True); self.value=nn.Conv2d(in_dim,value_dim,1,bias=False)
        self.attr_weight=nn.Parameter(torch.zeros(num_attributes,value_dim)); self.attr_bias=nn.Parameter(torch.zeros(num_attributes))
    def forward(self,fmap):
        a=torch.softmax(self.attn(fmap).flatten(2),dim=-1); v=self.value(fmap).flatten(2); pooled=torch.einsum('ban,bdn->bad',a,v)
        return (pooled*self.attr_weight.unsqueeze(0)).sum(-1)+self.attr_bias.unsqueeze(0)


class StripeAttributeHead(nn.Module):
    def __init__(self,in_dim,num_attributes=NUM_ATTRIBUTES,stripes=4):
        super().__init__(); self.stripes=int(stripes)
        self.attr_stripe_logits=nn.Parameter(torch.zeros(num_attributes,self.stripes))
        self.attr_weight=nn.Parameter(torch.zeros(num_attributes,in_dim)); self.attr_bias=nn.Parameter(torch.zeros(num_attributes))
    def forward(self,fmap):
        pooled=F.adaptive_avg_pool2d(fmap,(self.stripes,1)).squeeze(-1).transpose(1,2)
        weights=torch.softmax(self.attr_stripe_logits,dim=-1); attr_feat=torch.einsum('as,bsc->bac',weights,pooled)
        return (attr_feat*self.attr_weight.unsqueeze(0)).sum(-1)+self.attr_bias.unsqueeze(0)


class AttributePrototypeHead(nn.Module):
    def __init__(self,in_dim,prototype_dim=192,num_attributes=NUM_ATTRIBUTES,init_scale=8.0):
        super().__init__(); self.num_attributes=int(num_attributes); self.prototype_dim=int(prototype_dim)
        self.attn=nn.Conv2d(in_dim,self.num_attributes,1,bias=True)
        self.value=nn.Conv2d(in_dim,self.prototype_dim,1,bias=False)
        self.global_proj=nn.Linear(in_dim,self.prototype_dim,bias=False)
        self.local_gate_logits=nn.Parameter(torch.zeros(self.num_attributes))
        self.pos_prototypes=nn.Parameter(F.normalize(torch.randn(self.num_attributes,self.prototype_dim),dim=-1)*.10)
        self.neg_prototypes=nn.Parameter(F.normalize(torch.randn(self.num_attributes,self.prototype_dim),dim=-1)*.10)
        self.logit_scale=nn.Parameter(torch.tensor(float(init_scale)).log())
    def normalized_prototypes(self):
        return F.normalize(self.pos_prototypes.float(),dim=-1),F.normalize(self.neg_prototypes.float(),dim=-1)
    def forward(self,fmap,global_feat):
        a=torch.softmax(self.attn(fmap).flatten(2),dim=-1); v=self.value(fmap).flatten(2)
        local=torch.einsum('ban,bdn->bad',a,v); glob=self.global_proj(global_feat).unsqueeze(1).expand(-1,self.num_attributes,-1)
        gate=torch.sigmoid(self.local_gate_logits).view(1,self.num_attributes,1)
        feat=F.normalize(gate*local+(1-gate)*glob,dim=-1); pos,neg=self.normalized_prototypes()
        sp=torch.einsum('bad,ad->ba',feat.float(),pos); sn=torch.einsum('bad,ad->ba',feat.float(),neg)
        logits=self.logit_scale.exp().clamp(1.,30.)*(sp-sn)
        return logits.to(feat.dtype),feat



class RuntimeNet(nn.Module):
    def __init__(self,backbone='convnext_small',embed_dim=320):
        super().__init__()
        if backbone=='efficientnet_b0':
            base=efficientnet_b0(weights=None); feat=base.classifier[1].in_features
        elif backbone=='convnext_tiny':
            base=convnext_tiny(weights=None); feat=base.classifier[2].in_features
        elif backbone=='convnext_small':
            base=convnext_small(weights=None); feat=base.classifier[2].in_features
        elif backbone=='convnext_base':
            base=convnext_base(weights=None); feat=base.classifier[2].in_features
        elif backbone=='efficientnet_v2_s':
            base=efficientnet_v2_s(weights=None); feat=base.classifier[-1].in_features
        else:
            raise ValueError(backbone)
        self.features=base.features; self.avgpool=base.avgpool
        self.global_attr_head=nn.Linear(feat,NUM_ATTRIBUTES)
        self.spatial_attr_head=SpatialAttributeHead(feat,min(320,feat),NUM_ATTRIBUTES)
        self.stripe_attr_head=StripeAttributeHead(feat,NUM_ATTRIBUTES,4)
        self.prototype_head=AttributePrototypeHead(feat,min(256,max(128,embed_dim)),NUM_ATTRIBUTES,8.0)
        init_fusion=torch.zeros(NUM_ATTRIBUTES,4); init_fusion[:,3]=-1.10
        self.attr_fusion_logits=nn.Parameter(init_fusion)
        self.image_proj=nn.Sequential(nn.Linear(feat,embed_dim),nn.LayerNorm(embed_dim),nn.GELU(),nn.Dropout(.10),nn.Linear(embed_dim,embed_dim))
        self.query_encoder=QueryEncoder(embed_dim,.35)
    def encode_images(self,x):
        fmap=self.features(x); feat=torch.flatten(self.avgpool(fmap),1)
        g=self.global_attr_head(feat); s=self.spatial_attr_head(fmap); r=self.stripe_attr_head(fmap); p,_=self.prototype_head(fmap,feat)
        fusion=torch.softmax(self.attr_fusion_logits,dim=-1)
        logits=fusion[:,0].unsqueeze(0)*g+fusion[:,1].unsqueeze(0)*s+fusion[:,2].unsqueeze(0)*r+fusion[:,3].unsqueeze(0)*p
        return logits,F.normalize(self.image_proj(feat),dim=-1),p


class ABPRRuntime:
    """Self-contained Track-2 runtime used inside the Codabench submission."""
    def __init__(self,model_path: str|Path,config_path: str|Path|None=None):
        self.model_path=Path(model_path); self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models=[]; self.attribute_names=list(ATTRIBUTE_NAMES); self.height=384; self.width=192
        self.embedding_mix=0.; self.prototype_mix=0.; self.distance_kind='match'; self.positive_rarity_power=0.; self.negative_weight=1.; self.tta_flip=False
        self.attribute_weights=torch.ones(NUM_ATTRIBUTES); self.calibration_scale=torch.ones(NUM_ATTRIBUTES); self.calibration_bias=torch.zeros(NUM_ATTRIBUTES); self.priors=torch.full((NUM_ATTRIBUTES,),.5)
        self.batch_size=28 if self.device.type=='cuda' else 8; self.io_workers=min(8,max(2,os.cpu_count() or 2)); self._load()
    def _load(self):
        pkg=torch.load(self.model_path,map_location='cpu',weights_only=False)
        self.height=int(pkg.get('image_height',384)); self.width=int(pkg.get('image_width',192))
        self.embedding_mix=float(np.clip(pkg.get('embedding_mix',0.),0.,0.20)); self.prototype_mix=float(np.clip(pkg.get('prototype_mix',0.),0.,0.95)); self.distance_kind=str(pkg.get('distance_kind','match')).lower()
        self.positive_rarity_power=float(pkg.get('positive_rarity_power',0.)); self.negative_weight=float(pkg.get('negative_weight',1.)); self.tta_flip=bool(pkg.get('tta_flip',False))
        self.attribute_names=list(pkg.get('attribute_names') or ATTRIBUTE_NAMES)
        self.attribute_weights=torch.as_tensor(pkg.get('attribute_weights',[1.]*NUM_ATTRIBUTES),dtype=torch.float32).flatten().clamp_min(1e-6); self.attribute_weights/=self.attribute_weights.mean().clamp_min(1e-6)
        self.calibration_scale=torch.as_tensor(pkg.get('calibration_scale',[1.]*NUM_ATTRIBUTES),dtype=torch.float32).flatten().clamp(.25,4.)
        self.calibration_bias=torch.as_tensor(pkg.get('calibration_bias',[0.]*NUM_ATTRIBUTES),dtype=torch.float32).flatten().clamp(-3.,3.)
        self.priors=torch.as_tensor(pkg.get('priors',[.5]*NUM_ATTRIBUTES),dtype=torch.float32).flatten().clamp(1e-4,1-1e-4)
        if any(len(x)!=NUM_ATTRIBUTES for x in (self.attribute_names,self.attribute_weights,self.calibration_scale,self.calibration_bias,self.priors)):
            raise RuntimeError('Expected 40 attribute fields')
        for item in (pkg.get('models') or [pkg]):
            bb=item.get('backbone',pkg.get('backbone','convnext_small')); ed=int(item.get('embed_dim',pkg.get('embed_dim',320)))
            m=RuntimeNet(bb,ed)
            state=item.get('state') or item.get('inference_model_state') or item.get('ema_model_state') or item.get('model_state')
            allow=('features.','avgpool.','global_attr_head.','spatial_attr_head.','stripe_attr_head.','prototype_head.','attr_fusion_logits','image_proj.','query_encoder.')
            state={k:v for k,v in state.items() if k.startswith(allow)}
            missing,_=m.load_state_dict(state,strict=False)
            critical=[k for k in missing if k.startswith(('features.','global_attr_head.','spatial_attr_head.','stripe_attr_head.','prototype_head.','attr_fusion_logits'))]
            if critical: raise RuntimeError(f'Missing critical state keys: {critical[:10]}')
            self.models.append(m.to(self.device).eval())
        self.transform=transforms.Compose([transforms.Resize((self.height,self.width)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
        self.pool=ThreadPoolExecutor(max_workers=self.io_workers)
        if self.device.type=='cpu': torch.set_num_threads(min(8,max(1,os.cpu_count() or 1)))
    def _load_one(self,p):
        with Image.open(p) as im: return self.transform(im.convert('RGB'))
    @torch.inference_mode()
    def encode_gallery(self,samples):
        paths=[s.get('image_path') or s.get('image') or s.get('path') for s in samples]
        if any(p is None for p in paths): raise ValueError('Gallery item missing image_path/image/path')
        ps,zs=[],[]
        for st in range(0,len(paths),self.batch_size):
            ch=paths[st:st+self.batch_size]; xs=list(self.pool.map(self._load_one,ch)); x=torch.stack(xs).to(self.device,non_blocking=True)
            pp=[]; pr=[]; zz=[]
            for m in self.models:
                with torch.autocast(device_type=self.device.type,dtype=torch.float16,enabled=self.device.type=='cuda'):
                    logits,z,proto_logits=m.encode_images(x)
                    if self.tta_flip:
                        lf,zf,pf=m.encode_images(torch.flip(x,dims=[3])); logits=(logits+lf)*.5; proto_logits=(proto_logits+pf)*.5; z=F.normalize((z+zf)*.5,dim=-1)
                pp.append(torch.sigmoid(logits.float())); pr.append(torch.sigmoid(proto_logits.float())); zz.append(z.float())
            fused=torch.stack(pp).mean(0); proto=torch.stack(pr).mean(0)
            if self.prototype_mix>0: fused=(1-self.prototype_mix)*fused+self.prototype_mix*proto
            ps.append(fused.cpu()); zs.append(F.normalize(torch.stack(zz).mean(0),dim=-1).cpu())
        return torch.cat(ps,0),torch.cat(zs,0)
    @torch.inference_mode()
    def encode_queries(self,q):
        q=torch.as_tensor(q,dtype=torch.float32,device=self.device); z=[m.query_encoder(q).float() for m in self.models]
        return F.normalize(torch.stack(z).mean(0),dim=-1).cpu()
    def calibrate_probs(self,p):
        p=p.float().cpu().clamp(1e-6,1-1e-6); logits=torch.log(p/(1-p))
        return torch.sigmoid(logits*self.calibration_scale[None,:]+self.calibration_bias[None,:]).clamp(1e-6,1-1e-6)
    def _query_weights(self,q):
        valid=q>=0; pos=(q>.5)&valid; neg=(q<=.5)&valid; base=self.attribute_weights[None,:].expand(len(q),-1)
        rarity=torch.ones_like(base)
        if self.positive_rarity_power>0:
            r=(-torch.log(self.priors)).pow(self.positive_rarity_power); r=r/r.mean().clamp_min(1e-6); r=r.clamp(.35,3.5); rarity=r[None,:].expand_as(base)
        return pos.float()*base*rarity + neg.float()*base*self.negative_weight
    def attribute_distance(self,queries,gallery_probs):
        p=self.calibrate_probs(gallery_probs); q=torch.as_tensor(queries,dtype=torch.float32).cpu(); valid=q>=0; q01=torch.where(valid,q.clamp(0,1),torch.zeros_like(q)); wq=self._query_weights(q); den=wq.sum(1,keepdim=True).clamp_min(1e-6)
        if self.distance_kind=='match':
            sim=wq@(1-p).t()+(q01*wq)@(2*p-1).t(); d=-sim
        elif self.distance_kind=='l1':
            d=wq@p.t()+(q01*wq)@(1-2*p).t()
        elif self.distance_kind=='nll':
            neglog0=-torch.log1p(-p); delta=torch.log1p(-p)-torch.log(p); d=wq@neglog0.t()+(q01*wq)@delta.t()
        else: raise ValueError(self.distance_kind)
        return d/den
    def distance(self,queries,gallery_probs,gallery_emb=None):
        d=self.attribute_distance(queries,gallery_probs); mix=self.embedding_mix
        if mix<=0: return d
        qz=self.encode_queries(queries); gz=F.normalize(gallery_emb.float().cpu(),dim=-1); de=(1-qz@gz.t())*.5
        d=(d-d.mean(1,keepdim=True))/d.std(1,keepdim=True).clamp_min(1e-6); de=(de-de.mean(1,keepdim=True))/de.std(1,keepdim=True).clamp_min(1e-6)
        return (1-mix)*d+mix*de
    def rank(self,queries,samples):
        p,z=self.encode_gallery(samples); d=self.distance(queries,p,z).numpy(); return [np.argsort(r,kind='stable').tolist() for r in d],d
