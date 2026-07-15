import matplotlib.pyplot as plt

# 1. Configurações de estilo acadêmico (padrão IEEE/SBC)
plt.rcParams.update({
    'font.family': 'serif',       
    'font.size': 11,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'text.usetex': False          
})

# 2. Novos dados fornecidos para o cusco_dataset_1
data_new = {
    'ORB + KNN': (0.02254 * 1000, 0.5886, 0.0426, 'o', '#4472C4'),
    'SIFT + FLANN': (0.03335 * 1000, 0.1791, 0.0584, 's', '#ED7D31'),
    'SIFT + LightGlue': (0.11318 * 1000, 0.4774, 0.0479, 'D', '#A5A5A5'),
    'Superpoint + FLANN': (0.39841 * 1000, 0.2084, 0.0543, '^', '#70AD47'),
    'Superpoint + LightGlue': (0.46937 * 1000, 0.5197, 0.0458, 'v', '#9E480E'),
    'XFeat + FLANN': (0.07303 * 1000, 0.2272, 0.0511, 'p', '#FFC000'),
    'XFeat + LightGlue': (0.30344 * 1000, 0.4649, 0.0458, '*', '#FF0000')
}

# Criando a estrutura lado a lado (1 linha, 2 colunas)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2), dpi=300)

# ==========================================
# SUBPLOT 1: Erro Absoluto de Trajetória (ATE)
# ==========================================
for label, (time_ms, ate, rpe, marker, color) in data_new.items():
    size = 8
    ax1.scatter(time_ms, ate, marker=marker, color=color, s=size**2, 
                label=label, alpha=0.9, edgecolors='black', linewidths=0.5)

ax1.set_xlabel('Tempo Médio Total por Quadro (ms)')
ax1.set_ylabel('Erro Absoluto de Trajetória - ATE (m)')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.grid(True, linestyle='--', alpha=0.5, which='both')

# ==========================================
# SUBPLOT 2: Erro Relativo de Pose (RPE)
# ==========================================
for label, (time_ms, ate, rpe, marker, color) in data_new.items():
    size = 8
    ax2.scatter(time_ms, rpe, marker=marker, color=color, s=size**2, 
                label=label, alpha=0.9, edgecolors='black', linewidths=0.5)

ax2.set_xlabel('Tempo Médio Total por Quadro (ms)')
ax2.set_ylabel('Erro Relativo de Pose - RPE (m)')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.grid(True, linestyle='--', alpha=0.5, which='both')

# ==========================================
# LEGENDA GLOBAL E AJUSTES FINAIS
# ==========================================
handles, labels = ax1.get_legend_handles_labels()

fig.legend(handles, labels, loc='upper center', 
           bbox_to_anchor=(0.5, 1.05), 
           ncol=4,                     
           frameon=False)              

plt.tight_layout()
fig.subplots_adjust(top=0.85) 

# Salvando em PDF vetorial para o artigo
plt.savefig('cusco_dataset_1_performance_tradeoff_pt.pdf', bbox_inches='tight')
plt.show()