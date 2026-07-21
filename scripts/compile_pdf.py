import os
from PIL import Image

def compile_pdf():
    figures = [
        "figures/fig1.png",            # Assuming Fig 1 is empirical data or timeseries
        "figures/fig2_paired.png",     # Contour: Detection vs VE
        "figures/fig3_paired.png",     # Boxplots: Efficacy / PEP
        "figures/fig4_paired.png",     # Contour: Detection vs Delay
        "figures/fig5_paired.png",     # Tornado Sensitivity
        "figures/spaghetti.png",       # Cumulative deaths averted
    ]
    
    # Filter only existing figures
    existing = []
    for f in figures:
        if os.path.exists(f):
            existing.append(f)
        else:
            print(f"Warning: {f} not found, skipping.")
            
    if not existing:
        print("No figures found to compile.")
        return
        
    images = [Image.open(f).convert('RGB') for f in existing]
    
    pdf_path = "figures/all_figures.pdf"
    images[0].save(pdf_path, save_all=True, append_images=images[1:])
    print(f"Successfully created {pdf_path}")
    
    # Open on Mac
    os.system(f"open {pdf_path}")

if __name__ == "__main__":
    compile_pdf()
