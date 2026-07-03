import torch, torch.nn as nn, numpy as np
from scipy import stats
exec(open('kociu_3metrics.py').read().split("# ---------------- TEST DI SANITA'")[0])
# importo anche delta_phi_structured
def delta_phi_structured(net,X,labels):
    net.eval()
    with torch.no_grad(): H=net.hidden_rep(X).numpy()
    H=(H-H.mean(0))/(H.std(0)+1e-9)
    c0=H[labels==0]; c1=H[labels==1]; m0,m1=c0.mean(0),c1.mean(0)
    within=(np.mean(np.linalg.norm(c0-m0,axis=1))+np.mean(np.linalg.norm(c1-m1,axis=1)))/2
    between=np.linalg.norm(m0-m1)
    return float(between/(within+between)) if (within+between)>1e-9 else 0.0

# Raccolgo tutte e 4 le metriche + accuratezza su 50 reti
data={'acc':[],'dphi':[],'psi':[],'theta':[],'omega':[]}
for seed in range(50):
    torch.manual_seed(seed)
    Xs,Ys=make_spirals(seed=seed); lab=Ys.squeeze().numpy().astype(int)
    net=Net(); opt=torch.optim.Adam(net.parameters(),lr=0.01); lf=nn.BCELoss()
    n_ep=int(np.random.RandomState(seed).choice([0,20,50,100,200,500,1500,3000]))
    for _ in range(n_ep): opt.zero_grad(); lf(net(Xs),Ys).backward(); opt.step()
    with torch.no_grad(): acc=((net(Xs)>0.5).float()==Ys).float().mean().item()
    data['acc'].append(acc)
    data['dphi'].append(delta_phi_structured(net,Xs,lab))
    data['psi'].append(psi_order(net,Xs))
    data['theta'].append(theta_robust(net,Xs,Ys))
    data['omega'].append(omega_continuity(net,Xs))
for k in data: data[k]=np.array(data[k])

print("=== MATRICE DI CORRELAZIONE TRA LE 4 METRICHE + ACCURATEZZA (50 reti) ===\n")
keys=['acc','dphi','psi','theta','omega']
names={'acc':'Accuratezza','dphi':'Delta-phi','psi':'Psi','theta':'Theta','omega':'Omega'}
print(f"{'':<12}"+"".join(f"{names[k][:8]:>10}" for k in keys))
for k1 in keys:
    row=f"{names[k1][:11]:<12}"
    for k2 in keys:
        r=np.corrcoef(data[k1],data[k2])[0,1]
        row+=f"{r:>10.2f}"
    print(row)

print("\n=== TEST CHIAVE: le metriche predicono l'accuratezza MEGLIO INSIEME che da sole? ===")
from itertools import combinations
# regressione multipla: accuratezza predetta da combinazioni di metriche
def r2_multi(predictors):
    Xm=np.column_stack([data[p] for p in predictors])
    Xm=np.column_stack([np.ones(len(Xm)),Xm])
    beta,_,_,_=np.linalg.lstsq(Xm,data['acc'],rcond=None)
    pred=Xm@beta
    ss_res=np.sum((data['acc']-pred)**2); ss_tot=np.sum((data['acc']-data['acc'].mean())**2)
    return 1-ss_res/ss_tot
for combo in [['dphi'],['omega'],['psi'],['dphi','omega'],['dphi','psi','theta','omega']]:
    print(f"  R² usando {str(combo):<40} = {r2_multi(combo):.3f}")
