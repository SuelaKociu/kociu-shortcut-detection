exec(open('kociu_stresstest.py').read().split('print("="*78)')[0])
import torch, torch.nn as nn, numpy as np
print("="*70)
print("STRESS TEST (parte 3): ATTACCO AVVERSARIALE")
print("Una rete che bara MA viene addestrata anche a sembrare 'normale'")
print("nelle metriche del framework. Riesce a ingannarlo?")
print("="*70)

# Rete che bara (usa scorciatoia) ma con penalita' per non avere Psi troppo alto
# (cioe' cerca di nascondere la firma che il framework cerca)
def train_adversarial(net,X,Y,ep=1000,hide=True):
    opt=torch.optim.Adam(net.parameters(),lr=0.01); lf=nn.BCELoss()
    for _ in range(ep):
        opt.zero_grad()
        out=net(X); loss=lf(out,Y)
        if hide:
            H=net.hidden_rep(X)
            # penalita': vuole abbassare la correlazione interna (per non sembrare 'troppo ordinata')
            Hn=(H-H.mean(0))/(H.std(0)+1e-6)
            C=torch.corrcoef(Hn.T)
            off=C-torch.diag(torch.diag(C))
            loss=loss+0.5*torch.mean(off**2)  # spinge a de-correlare i neuroni
        loss.backward(); opt.step()

profiles=[]; is_cheat=[]
for seed in range(15):
    Xtr,Ytr=make_data(600,True,seed+10,1); lab=Ytr.squeeze().numpy().astype(int)
    # baro AVVERSARIALE (cerca di nascondersi)
    torch.manual_seed(seed); net=build_net(3,64,2,'tanh'); train_adversarial(net,Xtr,Ytr,hide=True)
    with torch.no_grad(): H=net.hidden_rep(Xtr).numpy()
    profiles.append([delta_phi_structured(H,lab),psi_order(H),omega_continuity(net,Xtr)]); is_cheat.append(1)
    # onesta normale
    Xcl,_=make_data(600,False,seed+10,1)
    torch.manual_seed(seed); net2=build_net(3,64,2,'tanh'); train(net2,Xcl,Ytr)
    with torch.no_grad(): H2=net2.hidden_rep(Xtr).numpy()
    profiles.append([delta_phi_structured(H2,lab),psi_order(H2),omega_continuity(net2,Xtr)]); is_cheat.append(0)

profiles=np.array(profiles); is_cheat=np.array(is_cheat)
from numpy.linalg import lstsq
correct=0
for i in range(len(profiles)):
    mask=np.ones(len(profiles),bool); mask[i]=False
    Xm=np.column_stack([np.ones(mask.sum()),profiles[mask]])
    beta,_,_,_=lstsq(Xm,is_cheat[mask],rcond=None)
    pred=1 if (np.array([1,*profiles[i]])@beta)>0.5 else 0
    if pred==is_cheat[i]: correct+=1
print(f"\nRilevamento del baro AVVERSARIALE (che si nasconde): {correct/len(profiles)*100:.1f}%")
print(f"\nProfili medi:")
print(f"  Baro-nascosto:  dphi={profiles[is_cheat==1,0].mean():.3f} psi={profiles[is_cheat==1,1].mean():.3f} omega={profiles[is_cheat==1,2].mean():.3f}")
print(f"  Onesta:         dphi={profiles[is_cheat==0,0].mean():.3f} psi={profiles[is_cheat==0,1].mean():.3f} omega={profiles[is_cheat==0,2].mean():.3f}")
