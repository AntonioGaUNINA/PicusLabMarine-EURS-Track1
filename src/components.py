"""Offline component loaders for the exact test1 teacher state dictionaries.

Use load_component(name, root='/path/to/extracted/bundle', device='cuda').
These return neural components, not the final posterior/SAM fusion pipeline.
"""
from pathlib import Path
import json,sys,math,importlib.util,hashlib
import torch
from torch import nn

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
class LoRALinear(nn.Module):
    def __init__(self,linear,rank=8):
        super().__init__();self.base=linear.requires_grad_(False);self.a=nn.Linear(linear.in_features,rank,bias=False);self.b=nn.Linear(rank,linear.out_features,bias=False)
    def forward(self,x):return self.base(x)+2*self.b(self.a(x))
class Embedding(nn.Module):
    def __init__(self,original):
        super().__init__();self.original=original;self.added=nn.Conv2d(6,original.proj.out_channels,3,padding=1,bias=False);self.extra=None
    def forward(self,rgb):return self.original(rgb)+self.added(self.extra)
class NativeRestormer(nn.Module):
    def __init__(self,root):
        super().__init__();arch=module(root/'assets/restormer/restormer_arch.py','test1_restormer')
        self.net=arch.Restormer(**json.loads((root/'assets/restormer/parameters.json').read_text()));self.net.patch_embed=Embedding(self.net.patch_embed);self.gates=nn.Parameter(torch.zeros(200))
    def forward(self,x,condition_id):
        self.net.patch_embed.extra=torch.cat([x[:,3:6],x[:,6:9].clamp(-2,3)],1)
        try:
            base=x[:,:3];return base+self.gates[condition_id].tanh()[:,None,None,None]*(self.net(base)-base)
        finally:self.net.patch_embed.extra=None
class MarineDINO(nn.Module):
    def __init__(self,root):
        super().__init__()
        from transformers import AutoConfig,AutoModel
        from unittest.mock import patch
        m=module(root/'assets/marine_dino/coralscapes_hub_model.py','test1_marine_dino')
        kwargs=m._segmenter_kwargs_from_hub(json.loads((root/'assets/marine_dino/config.json').read_text()))
        kwargs['encoder_id']=str(root/'assets/dinov3_base')
        def empty_encoder(*args,**kw):return AutoModel.from_config(AutoConfig.from_pretrained(root/'assets/dinov3_base',local_files_only=True),attn_implementation='sdpa')
        with patch.object(m.AutoModel,'from_pretrained',new=empty_encoder):self.model=m.Dinov3DPTSegmenter(**kwargs,image_processor=object())
        head=self.model.seg_head.head;found=[(i,x) for i,x in enumerate(head) if isinstance(x,nn.Conv2d) and x.out_channels==96];assert len(found)==1
        index,self.native=found[0];head[index]=nn.Identity()
        for i in range(index+1,len(head)):assert isinstance(head[i],nn.Upsample);head[i]=nn.Identity()
        for name,part in list(self.model.encoder.named_modules()):
            if isinstance(part,nn.Linear) and name.rsplit('.',1)[-1] in ['q_proj','v_proj']:
                parent,key=name.rsplit('.',1);setattr(self.model.encoder.get_submodule(parent),key,LoRALinear(part))
        self.eurs=nn.Conv2d(self.native.in_channels,5,1)
    def forward(self,x):
        enc=self.model.encoder(pixel_values=x,output_hidden_states=True);selected=self.model._select_hidden_states(enc.hidden_states);p=int(self.model.encoder.config.patch_size)
        fused=self.model.neck(selected,patch_height=x.shape[-2]//p,patch_width=x.shape[-1]//p);features=self.model.seg_head(fused)
        return self.eurs(features),self.native(features)
def load_component(name,root=None,device='cpu',verify_hash=True):
    root=Path(root) if root else Path(__file__).resolve().parent
    index=json.loads((root/'MODEL_INDEX.json').read_text());row=next(x for x in index['components'] if x['id']==name);path=root/row['file']
    if verify_hash:
        with path.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==row['sha256']
    if name=='restormer_native':model=NativeRestormer(root)
    elif name.startswith('dinov3_'):model=MarineDINO(root)
    elif name.startswith('mask2former_'):
        from transformers import Mask2FormerConfig,Mask2FormerForUniversalSegmentation
        cfg=Mask2FormerConfig.from_pretrained(root/'assets/mask2former',local_files_only=True);cfg.num_labels=5;cfg.output_auxiliary_logits=True;cfg.id2label=dict(enumerate(['background','coral','fish','elasmobranch','other']));cfg.label2id={v:k for k,v in cfg.id2label.items()};model=Mask2FormerForUniversalSegmentation(cfg)
    elif name=='drunet':
        sys.path.insert(0,str(root/'assets/drunet'));from models.network_unet import UNetRes
        model=UNetRes(in_nc=4,out_nc=3,nc=[64,128,256,512],nb=4,act_mode='R',downsample_mode='strideconv',upsample_mode='convtranspose')
    elif name=='depth_anything_v2_small':
        from transformers import AutoConfig,AutoModelForDepthEstimation
        model=AutoModelForDepthEstimation.from_config(AutoConfig.from_pretrained(root/'assets/depth_anything',local_files_only=True))
    elif name=='sam3':
        sys.path.insert(0,str(root/'assets/sam3_source'));from sam3 import model_builder
        model=model_builder.build_sam3_image_model(device=device,checkpoint_path=None,load_from_HF=False,enable_inst_interactivity=False,compile=False)
    else:raise ValueError(name)
    ck=torch.load(path,map_location='cpu',weights_only=True,mmap=True);model.load_state_dict(ck['state_dict'],strict=True)
    return model.to(device).eval().requires_grad_(False)
