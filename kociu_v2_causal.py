"""
FRAMEWORK KOCIU - RICOSTRUZIONE RIGOROSA - Delta-phi CAUSALE
Misura l'interdipendenza come INFLUENZA CAUSALE tra neuroni:
perturbo ogni neurone e misuro quanto cambia il resto del sistema.
Niente correlazioni passive (che il caso gonfia con pochi dati).
Niente target predefiniti. Test di sanita' PRIMA. Controllo casuale SEMPRE.
"""
import torch
import torch.nn as nn
import numpy as np

torch.manual_seed(0); np.random.seed(0)

# ---- COMPITO con dati a sufficienza: classificazione a 2 spirali ----
# (problema classico non-lineare, 400 punti invece di 4)
def make_spirals(n=200):
    n_per = n // 2
    theta = np.sqrt(np.random.rand(n_per)) * 3 * np.pi
    r_a = theta; xa = np.stack([r_a*np.cos(theta), r_a*np.sin(theta)],1) + np.random.randn(n_per,2)*0.3
    r_b = theta; xb = np.stack([-r_b*np.cos(theta), -r_b*np.sin(theta)],1) + np.random.randn(n_per,2)*0.3
    X = np.concatenate([xa,xb]).astype(np.float32)
    X = (X - X.mean(0)) / X.std(0)
    Y = np.concatenate([np.zeros(n_per), np.ones(n_per)]).astype(np.float32)
    return torch.tensor(X), torch.tensor(Y).unsqueeze(1)

X, Y = make_spirals(400)

class Net(nn.Module):
    def __init__(self, hidden=12):
        super().__init__()
        self.fc1 = nn.Linear(2, hidden); self.fc2 = nn.Linear(hidden, hidden); self.fc3 = nn.Linear(hidden,1)
        self.act = nn.Tanh()
    def hidden_rep(self, x):
        h1 = self.act(self.fc1(x)); h2 = self.act(self.fc2(h1)); return h2
    def forward(self, x):
        return torch.sigmoid(self.fc3(self.hidden_rep(x)))

# ---- DELTA-PHI CAUSALE ----
# Per ogni neurone i del layer nascosto: perturbo la sua attivazione (+epsilon)
# e misuro quanto cambia l'OUTPUT del sistema. Media su tutti gli input.
# Interdipendenza = quanto il sistema dipende collettivamente dai suoi neuroni
# in modo NON ridondante (se spegnere un neurone qualsiasi cambia molto, c'e' integrazione).
def delta_phi_causal(net, X, eps=0.5):
    net.eval()
    with torch.no_grad():
        h = net.hidden_rep(X)                # attivazioni nascoste [N x H]
        base_out = torch.sigmoid(net.fc3(h)) # output base
        H = h.shape[1]
        effects = []
        for i in range(H):
            h_pert = h.clone()
            h_pert[:, i] += eps              # perturbo SOLO il neurone i
            out_pert = torch.sigmoid(net.fc3(h_pert))
            effect_i = torch.mean(torch.abs(out_pert - base_out)).item()
            effects.append(effect_i)
        effects = np.array(effects)
        # Integrazione: quanto l'influenza e' DISTRIBUITA (tutti contribuiscono)
        # invece di concentrata in 1 solo neurone. Usiamo 1 - Gini-like concentration.
        if effects.sum() < 1e-9:
            return 0.0, effects
        p = effects / effects.sum()
        # entropia normalizzata della distribuzione di influenza: alta = integrazione distribuita
        entropy = -np.sum(p * np.log(p + 1e-12)) / np.log(H)
        return float(entropy), effects

# ---- TEST DI SANITA' ----
print("=== TEST DI SANITA' (regola 2) ===")
# A. Sistema dove conta UN solo neurone (no integrazione): output = solo neurone 0
class SingleNet(nn.Module):
    def __init__(self, hidden=12):
        super().__init__()
        self.fc1=nn.Linear(2,hidden); self.fc2=nn.Linear(hidden,hidden)
        self.fc3=nn.Linear(hidden,1); self.act=nn.Tanh()
        with torch.no_grad():
            self.fc3.weight.zero_(); self.fc3.weight[0,0]=5.0; self.fc3.bias.zero_()
    def hidden_rep(self,x): return self.act(self.fc2(self.act(self.fc1(x))))
    def forward(self,x): return torch.sigmoid(self.fc3(self.hidden_rep(x)))
single = SingleNet()
e_single,_ = delta_phi_causal(single, X)
print(f"A. Output dipende da 1 SOLO neurone:  delta_phi = {e_single:.4f}  (atteso BASSO)")

# B. Sistema dove tutti i neuroni contano uguale (max integrazione distribuita)
class EqualNet(SingleNet):
    def __init__(self,hidden=12):
        super().__init__(hidden)
        with torch.no_grad():
            self.fc3.weight.fill_(1.0); self.fc3.bias.zero_()
equal = EqualNet()
e_equal,_ = delta_phi_causal(equal, X)
print(f"B. Tutti i neuroni contano uguale:    delta_phi = {e_equal:.4f}  (atteso ALTO)")
print()

# ---- ESPERIMENTO ----
print("=== ESPERIMENTO: 3 condizioni + controllo ===\n")
# 1. casuale
net_rand = Net()
with torch.no_grad():
    acc_rand = (((net_rand(X)>0.5).float())==Y).float().mean().item()
dphi_rand,_ = delta_phi_causal(net_rand, X)

# 2. allenata
net = Net(); opt=torch.optim.Adam(net.parameters(),lr=0.01); lf=nn.BCELoss()
for ep in range(3000):
    opt.zero_grad(); loss=lf(net(X),Y); loss.backward(); opt.step()
with torch.no_grad():
    acc_tr = (((net(X)>0.5).float())==Y).float().mean().item()
dphi_tr, effects_tr = delta_phi_causal(net, X)

# 3. allenata + rumore nei pesi (degradata)
net_deg = Net(); net_deg.load_state_dict(net.state_dict())
with torch.no_grad():
    for p in net_deg.parameters(): p += torch.randn_like(p)*0.8
    acc_deg = (((net_deg(X)>0.5).float())==Y).float().mean().item()
dphi_deg,_ = delta_phi_causal(net_deg, X)

print(f"{'Condizione':<34}{'Accuratezza':<14}{'Delta-phi causale'}")
print("-"*64)
print(f"{'1. CASUALE (non allenata)':<34}{acc_rand*100:>5.1f}%{'':<8}{dphi_rand:.4f}")
print(f"{'2. ALLENATA (impara spirali)':<34}{acc_tr*100:>5.1f}%{'':<8}{dphi_tr:.4f}")
print(f"{'3. DEGRADATA (pesi + rumore)':<34}{acc_deg*100:>5.1f}%{'':<8}{dphi_deg:.4f}")
print()
print("Lettura onesta:")
print(f"  Sanita': single-neuron={e_single:.3f} < equal={e_equal:.3f}? "
      f"{'OK la metrica distingue' if e_single < e_equal else 'PROBLEMA: non distingue'}")
print(f"  Apprendimento: allenata({dphi_tr:.3f}) vs casuale({dphi_rand:.3f}) -> "
      f"{'AUMENTA' if dphi_tr>dphi_rand else 'NON aumenta'}")
print(f"  Degrado: degradata({dphi_deg:.3f}) vs allenata({dphi_tr:.3f}) -> "
      f"{'CALA' if dphi_deg<dphi_tr else 'NON cala'}")
