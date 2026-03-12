import json,os,random,math,time
import numpy as np
from pathlib import Path
from collections import defaultdict
import torch,torch.nn as nn,torch.nn.functional as F

BASE=Path(os.path.expanduser("~/Desktop/schemalabsai"))
P=BASE/"data"/"v1_precomputed"
CK=BASE/"checkpoints"/"schema_v1_production.pt"
if not CK.exists():CK=BASE/"checkpoints"/"schema_v1_production.pt"
with open(P/"metadata.json") as f:meta=json.load(f)
I2S={int(i):s for s,i in meta["ds_s2i"].items()}
NS=len(meta["ds_sectors"])
se=torch.load(P/"sector_emb_matrix.pt",weights_only=True)
d=640;nh=16;nl=128;ly=6;dr=0.1;sd=384;fp=7;mc=30;mr_=10;mi=10;mk=64

class MIDAS(nn.Module):
    def __init__(s):
        super().__init__();s.imputer=nn.Sequential(nn.Linear(d,d*4),nn.GELU(),nn.Dropout(dr),nn.Linear(d*4,d*2),nn.GELU(),nn.Linear(d*2,d))
        s.denoiser=nn.Sequential(nn.Linear(d,d*2),nn.GELU(),nn.Linear(d*2,1));s.norm=nn.LayerNorm(d)
    def forward(s,x,cm):
        mb=cm.unsqueeze(-1).bool().expand_as(x)
        for _ in range(mi):x=torch.where(mb,x,s.imputer(x))
        return s.norm(x),s.denoiser(x).squeeze(-1)

class CellProcessing(nn.Module):
    def __init__(s):
        super().__init__();s.value_proj=nn.Linear(1,d);s.numeric_embed=nn.Embedding(2,d);s.fp_proj=nn.Linear(fp,d)
        s.fusion=nn.Linear(d*3,d);s.norm=nn.LayerNorm(d);s.d_model=d
        pe=torch.zeros(mc,d);pos=torch.arange(0,mc,dtype=torch.float).unsqueeze(1)
        dv=torch.exp(torch.arange(0,d,2,dtype=torch.float)*(-math.log(10000.0)/d))
        pe[:,0::2]=torch.sin(pos*dv);pe[:,1::2]=torch.cos(pos*dv[:d//2]);s.register_buffer('pe',pe)
    def forward(s,cv,ci,df,cm):
        B,R,C=cv.shape;v=s.value_proj(cv.unsqueeze(-1));t=s.numeric_embed(ci.long())
        f=s.fp_proj(df).unsqueeze(1).expand(B,R,C,d);p=s.pe[:C].unsqueeze(0).unsqueeze(0).expand(B,R,-1,-1)
        return s.norm(s.fusion(torch.cat([v+p,t,f],dim=-1)))*cm.unsqueeze(1).unsqueeze(-1).float()

class SchemaProcessing(nn.Module):
    def __init__(s):
        super().__init__();s.proj=nn.Linear(sd,d)
        l=nn.TransformerEncoderLayer(d_model=d,nhead=nh,dim_feedforward=d*4,batch_first=True,dropout=dr,activation='gelu')
        s.transformer=nn.TransformerEncoder(l,num_layers=ly);s.norm=nn.LayerNorm(d)
    def forward(s,ce,cm):x=s.proj(ce);x=s.transformer(x,src_key_padding_mask=~cm);return s.norm(x)*cm.unsqueeze(-1).float()

class AxialAttentionLayer(nn.Module):
    def __init__(s):
        super().__init__();s.row_attn=nn.MultiheadAttention(d,nh,dropout=dr,batch_first=True);s.col_attn=nn.MultiheadAttention(d,nh,dropout=dr,batch_first=True)
        s.norm1=nn.LayerNorm(d);s.norm2=nn.LayerNorm(d);s.ffn=nn.Sequential(nn.Linear(d,d*4),nn.GELU(),nn.Dropout(dr),nn.Linear(d*4,d));s.norm3=nn.LayerNorm(d)
    def forward(s,x,cm):
        B,R,C,dd=x.shape;xr=x.reshape(B*R,C,dd);m=(~cm).unsqueeze(1).expand(B,R,C).reshape(B*R,C)
        a,_=s.row_attn(xr,xr,xr,key_padding_mask=m);x=x+s.norm1(a.view(B,R,C,dd))
        xc=x.permute(0,2,1,3).reshape(B*C,R,dd);a2,_=s.col_attn(xc,xc,xc);x=x+s.norm2(a2.view(B,C,R,dd).permute(0,2,1,3))
        return x+s.norm3(s.ffn(x))

class LocalReasoning(nn.Module):
    def __init__(s):super().__init__();s.layers=nn.ModuleList([AxialAttentionLayer() for _ in range(ly)])
    def forward(s,x,cm):
        for l in s.layers:x=l(x,cm)
        return x

class PerceiverLayer(nn.Module):
    def __init__(s):
        super().__init__();s.cross_attn=nn.MultiheadAttention(d,nh,dropout=dr,batch_first=True);s.self_attn=nn.MultiheadAttention(d,nh,dropout=dr,batch_first=True)
        s.norm1=nn.LayerNorm(d);s.norm2=nn.LayerNorm(d);s.ffn=nn.Sequential(nn.Linear(d,d*4),nn.GELU(),nn.Dropout(dr),nn.Linear(d*4,d));s.norm3=nn.LayerNorm(d)
    def forward(s,la,kv,m):a,_=s.cross_attn(la,kv,kv,key_padding_mask=m);la=la+s.norm1(a);a2,_=s.self_attn(la,la,la);la=la+s.norm2(a2);return la+s.norm3(s.ffn(la))

class GlobalReasoning(nn.Module):
    def __init__(s):super().__init__();s.latents=nn.Parameter(torch.randn(nl,d)*0.02);s.layers=nn.ModuleList([PerceiverLayer() for _ in range(ly)]);s.norm=nn.LayerNorm(d)
    def forward(s,x,cm):
        B,R,C,dd=x.shape;fl=x.reshape(B,R*C,dd);m=~cm.unsqueeze(1).expand(B,R,C).reshape(B,R*C);la=s.latents.unsqueeze(0).expand(B,-1,-1)
        for l in s.layers:la=l(la,fl,m)
        return s.norm(la).mean(dim=1)

class SectorHead(nn.Module):
    def __init__(s):super().__init__();s.proj=nn.Sequential(nn.Linear(d*2,d),nn.GELU(),nn.Dropout(dr),nn.Linear(d,sd))
    def forward(s,g,sp,sem):return F.normalize(s.proj(torch.cat([g,sp],dim=-1)),dim=-1)@F.normalize(sem,dim=-1).t()*10

class ClassificationHead(nn.Module):
    def __init__(s):super().__init__();s.head=nn.Sequential(nn.Linear(d,d*2),nn.GELU(),nn.Dropout(dr),nn.Linear(d*2,d),nn.GELU(),nn.Linear(d,NS))
    def forward(s,x):return s.head(x)

class MCM(nn.Module):
    def __init__(s):super().__init__();s.mask_token=nn.Parameter(torch.randn(d)*0.02);s.predictor=nn.Sequential(nn.Linear(d,d*2),nn.GELU(),nn.Dropout(dr),nn.Linear(d*2,1));s.mask_ratio=0.15

class MIRAS(nn.Module):
    def __init__(s):
        super().__init__();s.huber_bias=nn.Parameter(torch.zeros(d));s.huber_delta=1.0;s.retention_gate=nn.Sequential(nn.Linear(d,d),nn.Sigmoid())
        s.gd_lr=nn.Parameter(torch.tensor(0.01));s.eta=nn.Parameter(torch.ones(d));s.delta_param=nn.Parameter(torch.zeros(d));s.alpha=nn.Parameter(torch.ones(d)*0.5)
        s.low_rank_down=nn.Linear(d,mk,bias=False);s.low_rank_up=nn.Linear(mk,d,bias=False);s.gate=nn.Sequential(nn.Linear(d*2,d),nn.Sigmoid())
        s.l2_weight=nn.Parameter(torch.tensor(0.001));s.rms_norm=nn.LayerNorm(d)
    def forward(s,x):
        r=x;df=x-s.huber_bias;h=torch.where(df.abs()<=s.huber_delta,0.5*df**2,s.huber_delta*(df.abs()-0.5*s.huber_delta))
        x=x-0.01*h.sign()*h.abs().clamp(max=1.0);x=x*s.retention_gate(x);x=s.eta*x+s.delta_param
        lo=s.low_rank_up(s.low_rank_down(x));x=s.alpha*x+(1-s.alpha)*lo;g=s.gate(torch.cat([x,r],dim=-1));return s.rms_norm(g*x+(1-g)*r)

class SchemaV1Production(nn.Module):
    def __init__(s):
        super().__init__();s.midas=MIDAS();s.cell_proc=CellProcessing();s.schema_proc=SchemaProcessing()
        s.local_reason=LocalReasoning();s.global_reason=GlobalReasoning();s.sector_head=SectorHead()
        s.cls_head=ClassificationHead();s.mcm=MCM();s.miras=MIRAS()
    def forward(s,ce,cm,df,cv,cmask,ci,sem):
        sc=s.schema_proc(ce,cm);cl=s.cell_proc(cv,ci,df,cm);cl,_=s.midas(cl,cmask);cl=cl+sc.unsqueeze(1)
        lo=s.local_reason(cl,cm);B,R,C,dd=lo.shape;lo=s.miras(lo.reshape(B,R*C,dd)).reshape(B,R,C,dd)
        g=s.global_reason(lo,cm);sp=(sc*cm.unsqueeze(-1).float()).sum(1)/cm.sum(1,keepdim=True).float().clamp(min=1)
        return s.sector_head(g,sp,sem),s.cls_head(g)

print("Loading model...")
model=SchemaV1Production()
ckpt=torch.load(CK,map_location="cpu",weights_only=False)
res=model.load_state_dict(ckpt["model_state_dict"],strict=False)
print(f"Missing keys: {len(res.missing_keys)}, Unexpected: {len(res.unexpected_keys)}")
model.eval()
print(f"Epoch={ckpt.get('epoch','?')}, Acc={ckpt.get('accuracy',0):.1f}%")

print("Loading data...")
ce_a=torch.load(P/"col_embs.pt",weights_only=True);cm_a=torch.load(P/"col_mask.pt",weights_only=True)
df_a=torch.load(P/"dist_fps.pt",weights_only=True);cv_a=torch.load(P/"cell_values.pt",weights_only=True)
cmk_a=torch.load(P/"cell_mask.pt",weights_only=True);ci_a=torch.load(P/"cell_is_numeric.pt",weights_only=True)
lb_a=torch.load(P/"labels.pt",weights_only=True)

random.seed(42);idx=list(range(len(lb_a)));random.shuffle(idx)
test=idx[int(0.95*len(idx)):][:5000]
print(f"Testing 100 samples...\n")

cor=defaultdict(int);tot=defaultdict(int);t0=time.time()
with torch.no_grad():
    for ix in test:
        sl,cl=model(ce_a[ix].float().unsqueeze(0),cm_a[ix].unsqueeze(0).bool(),df_a[ix].float().unsqueeze(0),cv_a[ix].float().unsqueeze(0),cmk_a[ix].float().unsqueeze(0),ci_a[ix].float().unsqueeze(0),se)
        s=I2S[lb_a[ix].item()];tot[s]+=1
        if sl.argmax(-1).item()==lb_a[ix].item():cor[s]+=1

tc=sum(cor.values());tt=sum(tot.values())
print(f"OVERALL: {tc}/{tt} = {tc/tt*100:.1f}% [{time.time()-t0:.1f}s]\n")
print(f"{'Sector':35s} {'Cor':>4s} {'Tot':>4s} {'Acc':>7s}")
print("-"*53)
for s in sorted(tot.keys()):
    c=cor.get(s,0);t=tot[s];print(f"{s:35s} {c:4d} {t:4d} {c/t*100:6.1f}%")
