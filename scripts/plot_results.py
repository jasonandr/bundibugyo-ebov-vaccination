import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import time
import re
import os

def plot_heatmap():
    # Read the ring vaccination results
    df = pd.read_csv("results_ring.csv", index_col=0)
    
    # We want to format the axes as percentages
    y_vals = df.index.values # coverages
    x_vals = df.columns.astype(float).values # efficacies
    
    df.index = [f"{int(float(i)*100)}%" for i in df.index]
    df.columns = [f"{int(float(c)*100)}%" for c in df.columns]
    
    plt.figure(figsize=(10, 8))
    # Plot the percentage of population infected
    ax = sns.heatmap(df * 100, annot=True, fmt=".1f", cmap="YlOrRd", 
                     cbar_kws={'label': 'Final Outbreak Size (% of Population)'})
    
    # Overlay the analytical threshold for random mass vaccination
    # R_eff = R0 * (1 - p * VE) = 1 -> p = (1 - 1/R0) / VE
    R0 = 1.6
    threshold_p = (1.0 - 1.0/R0)
    
    # Calculate the boundary line coordinates in heatmap index space
    x_line = np.linspace(0.01, 0.5, 100) # avoid division by zero
    y_line = threshold_p / x_line
    
    # Convert data coordinates to heatmap axis coordinates
    # The heatmap x-axis corresponds to efficacies (0 to 0.5) over len(x_vals) bins
    # The heatmap y-axis corresponds to coverages (0 to 0.8) over len(y_vals) bins
    x_pixels = x_line / 0.5 * len(x_vals)
    y_pixels = y_line / 0.8 * len(y_vals)
    
    ax.plot(x_pixels, y_pixels, color='blue', linewidth=3, linestyle='--', label='Mass Vaccination Threshold (Analytical)')
    
    # Set limits to cut off line outside bounds
    ax.set_ylim(len(y_vals), 0)
    ax.set_xlim(0, len(x_vals))
    
    plt.title("Ebola Outbreak Size under Ring Vaccination\n(Stochastic Event-Driven Model)")
    plt.xlabel("Vaccine Efficacy (Reduction in Susceptibility)")
    plt.ylabel("Ring Vaccination Coverage (Probability of Contact Tracing)")
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    
    # Save with timestamp
    timestamp = int(time.time())
    img_name = f"heatmap_ring_vaccination_{timestamp}.png"
    img_path = os.path.join("figures", img_name)
    plt.savefig(img_path, dpi=150)
    print(f"Saved plot to {img_path}")
    
    # Regex swap in walkthrough.md
    walkthrough_path = "figures/walkthrough.md"
    if os.path.exists(walkthrough_path):
        with open(walkthrough_path, "r") as f:
            content = f.read()
            
        pattern = r"heatmap_[a-z_0-9]+\.png"
        
        if re.search(pattern, content):
            new_content = re.sub(pattern, img_name, content)
            with open(walkthrough_path, "w") as f:
                f.write(new_content)
            print("Updated walkthrough.md with new timestamped image.")
        else:
            print("Image placeholder not found in walkthrough.md.")
    else:
        print("walkthrough.md does not exist yet.")

if __name__ == "__main__":
    plot_heatmap()
