import numpy as np
from mmengine.evaluator import BaseMetric
from mmseg.registry import METRICS

def ex(s,k):
    v=s[k] if isinstance(s,dict) else getattr(s,k); v=v['data'] if isinstance(v,dict) and 'data' in v else (v.data if hasattr(v,'data') else v); v=v.detach().cpu().numpy() if hasattr(v,'detach') else v; return np.squeeze(np.asarray(v))
def prf(y,p,t):
    q=p>=t; tp=np.sum(q&(y==1)); fp=np.sum(q&(y==0)); fn=np.sum((~q)&(y==1)); pr=tp/(tp+fp) if tp+fp else 0.; rc=tp/(tp+fn) if tp+fn else 0.; f=2*pr*rc/(pr+rc) if pr+rc else 0.; return pr,rc,f
@METRICS.register_module()
class HierC1UnionBCEMetric(BaseMetric):
    def __init__(self,class1_threshold=.5,union_threshold=.5,ignore_index=255,collect_device='cpu',prefix=None): super().__init__(collect_device=collect_device,prefix=prefix); self.t1=float(class1_threshold); self.tu=float(union_threshold); self.ign=int(ignore_index)
    def process(self,data_batch,data_samples):
        for s in data_samples:
            gt=ex(s,'gt_sem_seg').astype(np.int64); p1=ex(s,'class1_prob'); pu=ex(s,'union_filament_prob'); v=gt!=self.ign; self.results.append(((gt==1)[v].astype(np.uint8),p1[v],((gt==1)|(gt==2))[v].astype(np.uint8),pu[v]))
    def compute_metrics(self,res):
        y1=np.concatenate([x[0].ravel() for x in res]); p1=np.concatenate([x[1].ravel() for x in res]); yu=np.concatenate([x[2].ravel() for x in res]); pu=np.concatenate([x[3].ravel() for x in res]); p,r,f=prf(y1,p1,self.t1); up,ur,uf=prf(yu,pu,self.tu); out=dict(Class1_Precision=p,Class1_Recall=r,Class1_F1=f,Union_Precision=up,Union_Recall=ur,Union_F1=uf)
        try:
            from sklearn.metrics import average_precision_score
            out['Class1_AUPRC']=float(average_precision_score(y1,p1)); out['Union_AUPRC']=float(average_precision_score(yu,pu))
        except Exception: out['Class1_AUPRC']=float('nan'); out['Union_AUPRC']=float('nan')
        return out
