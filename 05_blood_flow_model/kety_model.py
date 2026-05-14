import numpy as np
import matplotlib.pyplot as plt

def kety_model(time, Fp, Ps, ve=0.2):
    kep = Ps / ve
    E = 1 - np.exp(-Ps / max(Fp, 0.1))
    Ktrans = Fp * E
    Ct = Ktrans * np.exp(-kep * time)
    Ct = Ct / np.max(Ct) if np.max(Ct) > 0 else Ct
    return Ct

def plot_kety_curves():
    time = np.linspace(0, 5, 100)
    
    params = [
        {'Fp': 50,  'Ps': 10,  'color': 'blue',   'linestyle': '-',  'marker': 'o', 'label': 'Low flow (50)'},
        {'Fp': 150, 'Ps': 30,  'color': 'green',  'linestyle': '--', 'marker': 's', 'label': 'Medium (150)'},
        {'Fp': 300, 'Ps': 60,  'color': 'orange', 'linestyle': '-.', 'marker': '^', 'label': 'High (300)'},
        {'Fp': 500, 'Ps': 100, 'color': 'red',    'linestyle': ':',  'marker': 'D', 'label': 'Very high (500)'}
    ]
    
    plt.figure(figsize=(12, 7))
    
    for p in params:
        Ct = kety_model(time, Fp=p['Fp'], Ps=p['Ps'])
        plt.plot(time, Ct, color=p['color'], linestyle=p['linestyle'], 
                linewidth=2, marker=p['marker'], markersize=4, markevery=15, label=p['label'])
    
    plt.xlabel('Time (minutes)', fontsize=12)
    plt.ylabel('Normalized Concentration', fontsize=12)
    plt.title('Kety Model: Contrast Agent Washout', fontsize=14)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.ylim(-0.05, 1.05)
    plt.xlim(0, 5)
    
    for spine in plt.gca().spines.values():
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    plt.savefig('kety_model_curves.png', dpi=200, bbox_inches='tight')
    print("Saved: kety_model_curves.png")
    print("\n" + "="*50)
    print("KETY MODEL RESULTS")
    print("="*50)
    for p in params:
        Ct_end = kety_model(np.array([5]), Fp=p['Fp'], Ps=p['Ps'])[0]
        print(f"  {p['label']:20s}: end concentration = {Ct_end:.3f}")
    print("="*50)
    plt.show()

if __name__ == "__main__":
    plot_kety_curves()
