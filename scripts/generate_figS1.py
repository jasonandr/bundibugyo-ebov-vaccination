import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
from paths import result_path

def generate_figS1():
    raw = pd.read_csv(result_path("final_high_replicate_raw.csv"))

    # Baseline map
    baseline_df = raw[raw["scenario"] == "no_vaccination"]
    baseline_map = baseline_df.set_index(['seed', 'detection'])['deaths_percent'].to_dict()

    def da(row):
        key = (row['seed'], row['detection'])
        baseline_deaths = baseline_map.get(key, 45.4)
        if baseline_deaths == 0: return 0.0
        return max(0.0, (baseline_deaths - row['deaths_percent']) / baseline_deaths * 100.0)

    # We want standard VE vs scaled VE
    standard_ve = raw[(raw['scenario'] == 'vaccine_efficacy') & (raw['radius'] == 1)].copy()
    scaled_ve = raw[(raw['scenario'] == 'vaccine_efficacy_scaled_cfr') & (raw['radius'] == 1)].copy()

    if standard_ve.empty or scaled_ve.empty:
        print("Required scenarios not yet available in data.")
        return

    standard_ve['Averted'] = standard_ve.apply(da, axis=1)
    scaled_ve['Averted'] = scaled_ve.apply(da, axis=1)

    standard_ve['Assumption'] = 'Standard (Full CFR Reduction)'
    scaled_ve['Assumption'] = 'Diagnostic (Linearly Scaled CFR Reduction)'

    combined = pd.concat([standard_ve, scaled_ve], ignore_index=True)
    combined['VE (%)'] = (combined['level'].astype(float) * 100).astype(int)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    sns.pointplot(
        data=combined, 
        x='VE (%)', 
        y='Averted', 
        hue='Assumption',
        dodge=True,
        errorbar='ci', # Wait, CI might be tiny for 5000 replicates. Let's do IQR or just let seaborn default
        markers=['o', 's'],
        capsize=0.1,
        palette=['#34495e', '#e74c3c'],
        ax=ax
    )

    ax.set_title("Impact of Therapeutic Mortality Benefit on VE Sensitivity", fontsize=14, pad=15, fontweight='bold')
    ax.set_ylabel("Deaths averted vs. no vaccination (%)", fontsize=12)
    ax.set_xlabel("Prophylactic Vaccine Efficacy (%)", fontsize=12)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    png_path = f"figures/figS1_diagnostic_ve_{timestamp}.png"
    pdf_path = f"figures/figS1_diagnostic_ve_{timestamp}.pdf"

    plt.tight_layout()
    plt.savefig(png_path, bbox_inches='tight')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    
    print(f"Saved Supplementary Figure S1 to {png_path}")

    # Inject into walkthrough if it exists
    walkthrough_path = "/Users/jasonandrews/.gemini/antigravity/brain/97cc1b20-bd98-4767-bd8d-56aacb36b28d/walkthrough.md"
    if os.path.exists(walkthrough_path):
        import re
        with open(walkthrough_path, "r") as f: content = f.read()
        
        # We can append it if it doesn't exist
        if "### Figure S1: Diagnostic VE Sensitivity" not in content:
            content += f"\n\n### Figure S1: Diagnostic VE Sensitivity\n*(Demonstrates how the therapeutic benefit assumption flattens the VE gradient)*\n![Figure S1]({png_path})\n"
        else:
            content = re.sub(r"!\[Figure S1\].*?\.png\)", f"![Figure S1]({png_path})", content)
            
        with open(walkthrough_path, "w") as f: f.write(content)

if __name__ == "__main__":
    generate_figS1()
