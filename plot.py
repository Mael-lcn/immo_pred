import numpy as np

def psi_decharge(rho, rho_c, tau_chute):
    """
    Modèle de décharge résistive (RC).
    rho       : La variable densité (analogue au temps t)
    rho_c     : La densité critique (analogue à t=0, le début de la décharge)
    tau_chute : La constante de décroissance (analogue à tau = R*C)
    """
    # On s'assure que rho est supérieur à rho_c pour la décharge
    # Si rho < rho_c, la fonction renvoie 1 (avant le début de la chute)
    if rho < rho_c:
        return 1.0
    
    # Formule : exp( - (delta_rho) / tau )
    return np.exp(-(rho - rho_c) / tau_chute)

# Exemple d'utilisation avec un rapport défini
tau_valeur = 2.5 / 0.1  # Si vous gardez le même rapport que précédemment
resultat = psi_decharge(rho=30, rho_c=25, tau_chute=tau_valeur)
print(f"Psi: {resultat}")