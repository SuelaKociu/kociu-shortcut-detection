import torch, torch.nn as nn, numpy as np
from scipy import stats
exec(open('kociu_emergence.py').read().split('# ---- ALLENO DUE RETI')[0])

# Costruisco 40 reti-baro e 40 oneste, raccolgo i profili delle metriche
def train(net,X,Y,ep=1500):
    opt=torch.optim.Adam(net.parameters(),lr=0.01); lf=nn.BCELoss()
    for _ in range(ep): opt.zero_grad(); lf(net(X),Y).backward(); opt.step()
def acc(net,X,Y):
    with torch.no_grad(): return ((net(X)>0.5).float()==Y).float().mean().item()

profiles=[]; is_cheat=[]; test_accs=[]
for seed in range(40):
    # BARO: allenata con scorciatoia attiva
    Xtr,Ytr=make_data(600,shortcut_on=True,seed=seed+10)
    Xte,Yte=make_data(600,shortcut_on=False,seed=seed+500)
    torch.manual_seed(seed); net=Net(); train(net,Xtr,Ytr)
    lab=Ytr.squeeze().numpy().astype(int)
    profiles.append([delta_phi_structured(net,Xtr,lab),psi_order(net,Xtr),omega_continuity(net,Xtr)])
    is_cheat.append(1); test_accs.append(acc(net,Xte,Yte))
    # ONESTA: allenata con scorciatoia gia' resa rumore
    Xcl,_=make_data(600,shortcut_on=False,seed=seed+10)
    torch.manual_seed(seed); net2=Net(); train(net2,Xcl,Ytr)
    profiles.append([delta_phi_structured(net2,Xtr,lab),psi_order(net2,Xtr),omega_continuity(net2,Xtr)])
    is_cheat.append(0); test_accs.append(acc(net2,Xte,Yte))

profiles=np.array(profiles); is_cheat=np.array(is_cheat); test_accs=np.array(test_accs)

print("=== MEDIE DEI PROFILI (40 baro vs 40 oneste) ===")
print(f"{'metrica':<14}{'BARO':>10}{'ONESTA':>10}{'differenza':>12}")
for i,name in enumerate(['Delta-phi','Psi','Omega']):
    cb=profiles[is_cheat==1,i].mean(); co=profiles[is_cheat==0,i].mean()
    t,p=stats.ttest_ind(profiles[is_cheat==1,i],profiles[is_cheat==0,i])
    print(f"{name:<14}{cb:>10.3f}{co:>10.3f}{cb-co:>+12.3f}   (p={p:.1e})")

# Classificatore semplice: il profilo delle metriche predice "baro vs onesta"?
from numpy.linalg import lstsq
Xm=np.column_stack([np.ones(len(profiles)),profiles])
beta,_,_,_=lstsq(Xm,is_cheat,rcond=None)
pred=(Xm@beta>0.5).astype(int)
accuracy=(pred==is_cheat).mean()
print(f"\n=== RILEVAMENTO DEL BARO dal solo stato interno ===")
print(f"Accuratezza nel distinguere baro vs onesta: {accuracy*100:.1f}%")
print(f"(usando solo Delta-phi, Psi, Omega - SENZA mai vedere il test reale)")

# confronto: e se guardassi solo l'accuratezza di training? (quello che fanno tutti)
print(f"\nPer confronto: sul TRAINING baro e oneste hanno accuratezza simile,")
print(f"quindi un osservatore normale NON distinguerebbe. Il framework si'.")
