"""
FRAMEWORK KOCIU - RICOSTRUZIONE RIGOROSA DA ZERO
Metrica 1: Delta-phi come INTERDIPENDENZA misurata DALL'INTERNO

Principio: nessun target predefinito. Costruiamo una rete vera, la alleniamo
davvero su XOR, e misuriamo l'integrazione sulle ATTIVAZIONI REALI dei neuroni
nascosti mentre la rete elabora. Confrontiamo tre condizioni e un controllo casuale.
"""
import torch
import torch.nn as nn
import numpy as np

torch.manual_seed(0)
np.random.seed(0)

# ---- 1. IL COMPITO: XOR (vero, non simulato) ----
X = torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
Y = torch.tensor([[0.],[1.],[1.],[0.]])

# ---- 2. LA RETE: piccola, ispezionabile nodo per nodo ----
class XORNet(nn.Module):
    def __init__(self, hidden=8):
        super().__init__()
        self.fc1 = nn.Linear(2, hidden)
        self.fc2 = nn.Linear(hidden, 1)
        self.act = nn.Tanh()
    def forward(self, x, return_hidden=False):
        h = self.act(self.fc1(x))
        out = torch.sigmoid(self.fc2(h))
        if return_hidden:
            return out, h
        return out

# ---- 3. METRICA DELTA-PHI (interdipendenza, misurata dall'interno) ----
# Definizione operativa onesta e testabile:
# Quanto i neuroni nascosti co-variano insieme quando il sistema elabora tutti gli input.
# Se i neuroni lavorano in modo interdipendente (integrato), le loro attivazioni
# sono correlate in modo strutturato. Misuriamo questo via la "forza" delle
# correlazioni fuori-diagonale tra le attivazioni dei neuroni nascosti.
def delta_phi(hidden_activations):
    # hidden_activations: matrice [n_input_patterns x n_neuroni]
    H = hidden_activations.detach().numpy()
    n_neurons = H.shape[1]
    if n_neurons < 2 or H.shape[0] < 2:
        return 0.0
    # correlazione tra neuroni (colonne)
    C = np.corrcoef(H.T)
    C = np.nan_to_num(C)
    iu = np.triu_indices(n_neurons, k=1)
    off_diag = C[iu]
    # interdipendenza = media del valore assoluto delle correlazioni fuori diagonale
    return float(np.mean(np.abs(off_diag)))

# ---- 4. TEST DI SANITA' (regola 2: testare su casi noti PRIMA) ----
print("=== TEST DI SANITA' DELLA METRICA (prima di usarla) ===")
# Caso A: neuroni perfettamente indipendenti (rumore puro) -> deve dare ~0
indep = np.random.normal(0,1,(4, 8))
print(f"A. Neuroni indipendenti (rumore):       delta_phi = {delta_phi(torch.tensor(indep)):.4f}  (atteso ~0, basso)")
# Caso B: neuroni perfettamente accoppiati (tutti uguali) -> deve dare ~1
coupled = np.tile(np.random.normal(0,1,(4,1)), (1,8))
print(f"B. Neuroni identici (max interdipend.): delta_phi = {delta_phi(torch.tensor(coupled)):.4f}  (atteso ~1, alto)")
print()

# ---- 5. ESPERIMENTO VERO: 3 CONDIZIONI ----
print("=== ESPERIMENTO: integrazione misurata dall'interno ===\n")

# CONDIZIONE 1: rete CASUALE (non allenata) - controllo
net_random = XORNet(hidden=8)
with torch.no_grad():
    _, h_random = net_random(X, return_hidden=True)
dphi_random = delta_phi(h_random)
# accuratezza della rete casuale
with torch.no_grad():
    pred_random = (net_random(X) > 0.5).float()
acc_random = (pred_random == Y).float().mean().item()

# CONDIZIONE 2: rete ALLENATA davvero
net = XORNet(hidden=8)
opt = torch.optim.Adam(net.parameters(), lr=0.05)
loss_fn = nn.BCELoss()
for epoch in range(2000):
    opt.zero_grad()
    out = net(X)
    loss = loss_fn(out, Y)
    loss.backward()
    opt.step()
with torch.no_grad():
    out, h_trained = net(X, return_hidden=True)
    pred_trained = (out > 0.5).float()
acc_trained = (pred_trained == Y).float().mean().item()
dphi_trained = delta_phi(h_trained)

# CONDIZIONE 3: rete allenata ma ABLATA (spengo metà dei neuroni nascosti)
with torch.no_grad():
    _, h_full = net(X, return_hidden=True)
    h_ablated = h_full.clone()
    h_ablated[:, 4:] = 0.0  # spengo 4 neuroni su 8
dphi_ablated = delta_phi(h_ablated)

print(f"{'Condizione':<32}{'Accuratezza XOR':<18}{'Delta-phi (integrazione)'}")
print(f"{'-'*72}")
print(f"{'1. Rete CASUALE (non allenata)':<32}{acc_random*100:>6.1f}%{'':<11}{dphi_random:.4f}")
print(f"{'2. Rete ALLENATA (impara XOR)':<32}{acc_trained*100:>6.1f}%{'':<11}{dphi_trained:.4f}")
print(f"{'3. Rete allenata ma ABLATA':<32}{'(rotta)':<18}{dphi_ablated:.4f}")
print()
print("Lettura onesta:")
print(f"  - La rete allenata raggiunge {acc_trained*100:.0f}% su XOR: ha imparato davvero.")
print(f"  - Delta-phi allenata ({dphi_trained:.3f}) vs casuale ({dphi_random:.3f}): "
      f"{'AUMENTA' if dphi_trained>dphi_random else 'NON aumenta'} con l'apprendimento.")
print(f"  - Ablando neuroni, delta-phi {'cala' if dphi_ablated<dphi_trained else 'NON cala'} "
      f"({dphi_ablated:.3f}).")
