import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import time

def draw_box(ax, center, text, color):
    x, y = center
    width, height = 3.0, 1.2
    box = patches.Rectangle((x - width/2, y - height/2), width, height, linewidth=2, edgecolor='black', facecolor=color, zorder=2)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=12, fontweight='bold', zorder=3)

def draw_arrow(ax, start, end, label=""):
    x1, y1 = start
    x2, y2 = end
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", lw=2, color="black", shrinkA=10, shrinkB=10), zorder=1)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my+0.3, label, ha='center', va='center', fontsize=10, fontweight='bold', color='darkred')

def generate_history_schema():
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Disease States
    S_pos = (2, 6)
    E_pos = (6, 6)
    I_pos = (10, 6)
    R_pos = (10, 8)
    D_pos = (10, 4)
    
    # Vaccination States
    V_PEND_pos = (2, 3)
    V_pos = (6, 3)
    
    draw_box(ax, S_pos, "Susceptible (S)", "lightblue")
    draw_box(ax, E_pos, "Exposed (E)\n[Incubation: 8.5d]", "lightyellow")
    draw_box(ax, I_pos, "Infectious (I)", "salmon")
    draw_box(ax, R_pos, "Recovered (R)", "lightgreen")
    draw_box(ax, D_pos, "Dead (D)", "gray")
    
    draw_box(ax, V_PEND_pos, "Vaccine Administered\n(V_PENDING)", "plum")
    draw_box(ax, V_pos, "Immune (V)", "violet")
    
    # Disease Flow
    draw_arrow(ax, S_pos, E_pos, "Infection")
    draw_arrow(ax, E_pos, I_pos, "Onset")
    draw_arrow(ax, I_pos, R_pos, "Survival")
    draw_arrow(ax, I_pos, D_pos, "Fatality (CFR 56%)")
    
    # Vaccination Flow
    draw_arrow(ax, S_pos, V_PEND_pos, "Ring Vaccination\n(Tracing Delay: 2d)")
    draw_arrow(ax, V_PEND_pos, V_pos, "Immune Delay\n(10d)")
    
    # Breakthrough Infection Flow
    # V_PENDING can still be exposed
    draw_arrow(ax, (V_PEND_pos[0]+1.5, V_PEND_pos[1]), (E_pos[0]-0.5, E_pos[1]-0.6), "Exposure\nbefore immunity")
    
    # Reduced CFR Note
    ax.text(8.0, 4.5, "Vaccinated Breakthrough CFR: 25%", fontsize=11, fontweight='bold', color='darkgreen', bbox=dict(facecolor='white', edgecolor='darkgreen', boxstyle='round,pad=0.5'))
    
    ax.set_xlim(0, 12)
    ax.set_ylim(2, 9)
    ax.axis('off')
    
    plt.title("Model Schema: Disease Natural History & Vaccination Timeline", fontsize=15, fontweight='bold')
    
    timestamp = int(time.time())
    img_path = os.path.join("figures", f"schema_history_{timestamp}.png")
    plt.savefig(img_path, dpi=200, bbox_inches='tight')
    print(f"Saved History Schema to {img_path}")

if __name__ == "__main__":
    generate_history_schema()
