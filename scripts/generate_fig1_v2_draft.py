import matplotlib.pyplot as plt
import matplotlib.patches as patches
import datetime
import numpy as np

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), dpi=300)

# =====================================================================
# Panel A: Network Schematic with Delay Annotations
# =====================================================================
ax1.set_title("A. Ring Vaccination Logistical Flow", fontsize=18, fontweight='bold', pad=20)
ax1.set_xlim(-5, 5)
ax1.set_ylim(-5, 5)
ax1.axis('off')

# Center Node (Index)
index_circle = patches.Circle((0, 0), 0.3, color='#cc0000', zorder=5)
ax1.add_patch(index_circle)
ax1.text(0, -0.6, "Index Case\n(Detected)", ha='center', va='top', fontsize=12, fontweight='bold')

# Ring 1 Boundary
ring1_boundary = patches.Circle((0, 0), 2.0, fill=False, edgecolor='#0044cc', linestyle='-', linewidth=3, alpha=0.5)
ax1.add_patch(ring1_boundary)

# Ring 1 Nodes (Household)
theta1 = np.linspace(0, 2*np.pi, 6, endpoint=False)
for t in theta1:
    x, y = 1.5 * np.cos(t), 1.5 * np.sin(t)
    ax1.add_patch(patches.Circle((x, y), 0.2, color='#0044cc', zorder=4))
ax1.text(-1.5, 2.2, "Ring 1\n(Household Contacts)", color='#0044cc', fontsize=14, fontweight='bold', ha='center')

# Ring 2 Boundary
ring2_boundary = patches.Circle((0, 0), 4.0, fill=False, edgecolor='#e68a00', linestyle='--', linewidth=3, alpha=0.5)
ax1.add_patch(ring2_boundary)

# Ring 2 Nodes (Community)
theta2 = np.linspace(0, 2*np.pi, 12, endpoint=False)
for t in theta2:
    x, y = 3.5 * np.cos(t), 3.5 * np.sin(t)
    ax1.add_patch(patches.Circle((x, y), 0.2, color='#e68a00', zorder=4))
ax1.text(-3.5, 4.2, "Ring 2\n(Contacts of Contacts)", color='#e68a00', fontsize=14, fontweight='bold', ha='center')

# Arrows indicating Logistical Delay
# Index to Ring 1
ax1.annotate("Immediate Tracing\n(e.g., 2 Days)", xy=(1.0, 1.0), xytext=(0.2, 0.2),
             arrowprops=dict(facecolor='#0044cc', edgecolor='#0044cc', shrink=0.05, width=3, headwidth=10),
             fontsize=12, fontweight='bold', color='#0044cc', ha='left', va='bottom', rotation=45)

# Ring 1 to Ring 2
ax1.annotate("Secondary Tracing\n(+2 Days Delay)", xy=(3.0, 0), xytext=(1.7, 0),
             arrowprops=dict(facecolor='#e68a00', edgecolor='#e68a00', shrink=0.05, width=3, headwidth=10, linestyle='--'),
             fontsize=12, fontweight='bold', color='#e68a00', ha='center', va='bottom')

# =====================================================================
# Panel B: Timeline (Gantt Chart)
# =====================================================================
ax2.set_title("B. The Speed Trade-off: Vaccination vs. Transmission", fontsize=18, fontweight='bold', pad=20)
ax2.set_xlim(0, 14)
ax2.set_ylim(-0.5, 3.5)
ax2.set_yticks([0, 1, 2, 3])
ax2.set_yticklabels(["Transmission\nEvent", "Base Case\n(No Intervention)", "Ring 1\n(Household)", "Ring 2\n(Community)"][::-1], fontsize=12, fontweight='bold')
ax2.set_xlabel("Days Since Index Case Onset", fontsize=14)
ax2.grid(True, axis='x', linestyle='--', alpha=0.5)

# Y-positions
y_trans, y_base, y_r1, y_r2 = 3, 2, 1, 0

# Timeline 1: The Virus
ax2.plot([0, 7], [y_trans, y_trans], color='grey', linewidth=4, zorder=1)
ax2.scatter(0, y_trans, color='red', s=150, zorder=2, label="Index Onset")
ax2.scatter(7, y_trans, color='black', s=150, zorder=2, marker='X', label="Secondary Exposure")
ax2.text(3.5, y_trans+0.15, "Infectious Period", ha='center', va='bottom', fontsize=11, color='grey')

# Timeline 2: Base Case (Detection but no vaccine)
ax2.plot([0, 4], [y_base, y_base], color='black', linewidth=2, linestyle=':', zorder=1)
ax2.scatter(4, y_base, color='purple', s=150, zorder=2, marker='D', label="Detection & Isolation")
ax2.text(2, y_base+0.15, "Reporting Delay", ha='center', va='bottom', fontsize=11)

# Timeline 3: Ring 1 (Vaccine arrives before exposure)
ax2.plot([0, 4], [y_r1, y_r1], color='black', linewidth=2, linestyle=':', zorder=1)
ax2.plot([4, 6], [y_r1, y_r1], color='#0044cc', linewidth=4, zorder=1)
ax2.scatter(4, y_r1, color='purple', s=150, zorder=2, marker='D')
ax2.scatter(6, y_r1, color='#0044cc', s=150, zorder=2, marker='v', label="Vaccination")
ax2.text(5, y_r1+0.15, "Trace Delay (2d)", ha='center', va='bottom', fontsize=11, color='#0044cc')

# Highlight success
ax2.axvline(7, color='black', linestyle='--', alpha=0.3, zorder=0)
ax2.text(6.1, y_r1-0.2, "Protected!\n(Arrives before Day 7)", color='green', fontweight='bold', fontsize=10)

# Timeline 4: Ring 2 (Vaccine arrives after exposure)
ax2.plot([0, 4], [y_r2, y_r2], color='black', linewidth=2, linestyle=':', zorder=1)
ax2.plot([4, 6], [y_r2, y_r2], color='#0044cc', linewidth=4, zorder=1)
ax2.plot([6, 8], [y_r2, y_r2], color='#e68a00', linewidth=4, zorder=1)
ax2.scatter(4, y_r2, color='purple', s=150, zorder=2, marker='D')
ax2.scatter(6, y_r2, color='#0044cc', s=80, zorder=2, marker='o') # Ring 1 traced
ax2.scatter(8, y_r2, color='#e68a00', s=150, zorder=2, marker='v')
ax2.text(7, y_r2+0.15, "Secondary Trace (+2d)", ha='center', va='bottom', fontsize=11, color='#e68a00')

# Highlight failure
ax2.text(8.1, y_r2-0.2, "Too Late!\n(Arrives after Day 7)", color='red', fontweight='bold', fontsize=10)

ax2.legend(loc='upper left', fontsize=11, framealpha=1.0)

plt.tight_layout()
timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
filename = f"figures/fig1_v2_draft_{timestamp}.png"
plt.savefig(filename, bbox_inches='tight', dpi=300, facecolor='white')
print(f"Saved {filename}")
