import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def generate_g_functions(A11, A12, A21, A22):
    def get_up(p, q):
        return (p*q*A11 + p*(1-q)*A12 + (1-p)*q*A21 + (1-p)*(1-q)*A22)

    def g_PG(p, q):
        return (q*A11 + (1-q)*A12) - get_up(p, q)

    def g_PD(p, q):
        return (q*A21 + (1-q)*A22) - get_up(p, q)

    def g_KG(p, q):
        return -(p*A11 + (1-p)*A21) - (-get_up(p, q))

    def g_KD(p, q):
        return -(p*A12 + (1-p)*A22) - (-get_up(p, q))

    return g_PG, g_PD, g_KG, g_KD

# --- Configuration du jeu ---
# Tireur fort à gauche (A11=0.5 au lieu de 0)
g_PG, g_PD, g_KG, g_KD = generate_g_functions(0.5, 1, 1, 0)

def F_vec(x):
    p, q = x
    regrets = [g_PG(p, q), g_PD(p, q), g_KG(p, q), g_KD(p, q)]
    return sum(max(r, 0)**2 for r in regrets)

# --- Suivi de l'optimisation ---
historique = []
def callback_bfgs(xk):
    # On enregistre p et q à chaque étape
    historique.append(np.copy(xk))

# --- Lancement du BFGS ---
x0 = np.array([0.1, 0.1]) # Départ un peu excentré pour mieux voir la trajectoire
bounds = [(0, 1), (0, 1)]
historique.append(np.copy(x0)) # Ajout du point de départ

res = minimize(F_vec, x0, method='L-BFGS-B', bounds=bounds, callback=callback_bfgs)

# --- Affichage et Graphique ---
if res.success:
    p_opt, q_opt = res.x
    pts = np.array(historique)
    
    plt.figure(figsize=(10, 6))
    
    # 1. Dessiner le chemin parcouru
    plt.plot(pts[:, 0], pts[:, 1], 'ro-', label="Chemin du BFGS", markersize=4)
    plt.plot(pts[0, 0], pts[0, 1], 'go', label="Départ", markersize=8)
    plt.plot(p_opt, q_opt, 'bx', label="Équilibre (Nash)", markersize=10, markeredgewidth=3)

    # 2. Mise en forme
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.xlabel('Probabilité Tireur Gauche (p)')
    plt.ylabel('Probabilité Gardien Gauche (q)')
    plt.title(f'Convergence BFGS vers l\'Équilibre de Nash\nFinal: p={p_opt:.2f}, q={q_opt:.2f}')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Annotation pour donner du sens
    plt.annotate(f'Point Final\nF={res.fun:.1e}', xy=(p_opt, q_opt), xytext=(p_opt+0.1, q_opt+0.1),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1))

    plt.show()

    print(f"Convergence réussie en {len(pts)} itérations.")
    print(f"Résultat : p={p_opt:.4f}, q={q_opt:.4f}")