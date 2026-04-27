import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------
# DATA FROM TABLES
# ---------------------------------------------------------
states_static = ['Prior\n(T)', 'Intermediate\n(TA)', 'Posterior\n(TAM)']
states_transition = ['Abstract Gain\n(T -> TA)', 'Method Gain\n(TA -> TAM)']

# Panel A: Structural Accuracy (Minimum Cosine Distance)
dist_gpt = [0.2212, 0.1390, 0.1127]
dist_ds  = [0.2387, 0.1738, 0.1696]

# Panel B: Lexical Accuracy (Residual Ambiguity to GT in Bits)
res_amb_gpt = [0.8147, 0.3521, 0.3434]
res_amb_ds  = [0.8121, 0.2444, 0.1961]

# Panel C: Internal Lexical Ambiguity (Keyword Entropy in Bits)
ent_key_gpt = [4.101, 3.496, 1.203]
ent_key_ds  = [3.279, 0.947, 0.619]

# Panel D: Lexical Information Gain (JSD Shifts in Bits)
jsd_lex_gpt = [0.2934, 0.4615]
jsd_lex_ds  = [0.4837, 0.2486]

# ---------------------------------------------------------
# PLOTTING SETUP (2x2 Grid)
# ---------------------------------------------------------
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
#fig.suptitle("Cognitive Trajectories: Reasoning vs. Instruction Models", fontsize=16, fontweight='bold', y=0.96)

color_gpt = '#1f77b4' # Blue
color_ds = '#d62728'  # Red

ax1 = axs[0, 0] # Top-Left
ax2 = axs[0, 1] # Top-Right
ax3 = axs[1, 0] # Bottom-Left
ax4 = axs[1, 1] # Bottom-Right

# --- Panel A: Structural Accuracy (Cosine Distance) ---
ax1.plot(states_static, dist_gpt, marker='o', linewidth=2.5, color=color_gpt, label='GPT-OSS (Instruction)')
ax1.plot(states_static, dist_ds, marker='s', linewidth=2.5, color=color_ds, label='DeepSeek (Reasoning)')
#ax1.axhline(y=0.15, color='gray', linestyle='--', alpha=0.8, label='Accuracy Threshold (0.15)')

ax1.set_ylabel('Cosine Distance\n(Lower = Better)', fontsize=12)
ax1.set_title('A. Structural Alignment to Ground Truth', fontsize=12, fontweight='bold', loc='left')
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend()

# --- Panel B: Lexical Accuracy (Residual Ambiguity) ---
ax2.plot(states_static, res_amb_gpt, marker='o', linewidth=2.5, color=color_gpt)
ax2.plot(states_static, res_amb_ds, marker='s', linewidth=2.5, color=color_ds)

ax2.set_ylabel('Divergence (Bits)\n(Lower = Better Match)', fontsize=12)
ax2.set_title('B. Lexical Alignment to Ground Truth', fontsize=12, fontweight='bold', loc='left')
ax2.grid(True, linestyle=':', alpha=0.6)

# --- Panel C: Internal Lexical Ambiguity (Keyword Entropy) ---
ax3.plot(states_static, ent_key_gpt, marker='o', linewidth=2.5, color=color_gpt)
ax3.plot(states_static, ent_key_ds, marker='s', linewidth=2.5, color=color_ds)

ax3.set_ylabel('Entropy (Bits)\n(Lower = Higher Certainty)', fontsize=12)
ax3.set_title('C. Internal Lexical Certainty', fontsize=12, fontweight='bold', loc='left')
ax3.grid(True, linestyle=':', alpha=0.6)

# --- Panel D: Lexical Information Gain (JSD Transitions) ---
x = np.arange(len(states_transition))
width = 0.35

ax4.bar(x - width/2, jsd_lex_gpt, width, color=color_gpt, alpha=0.8, label='GPT-OSS')
ax4.bar(x + width/2, jsd_lex_ds, width, color=color_ds, alpha=0.8, label='DeepSeek')

ax4.set_xticks(x)
ax4.set_xticklabels(states_transition)
ax4.set_ylabel('Divergence (Bits)\n(Higher = Larger Shift)', fontsize=12)
ax4.set_title('D. Lexical Information Gain', fontsize=12, fontweight='bold', loc='left')
ax4.grid(axis='y', linestyle=':', alpha=0.6)
ax4.legend()

# Adjust layout and show
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig('cognitive_trajectories_comparison.pdf', dpi=300, bbox_inches='tight')
plt.show()
