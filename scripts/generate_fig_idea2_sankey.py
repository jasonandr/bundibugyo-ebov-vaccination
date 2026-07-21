import plotly.graph_objects as go
import datetime
import os

print("Generating Idea 2: Sankey Diagram...")

# Define nodes
labels = [
    "Radius 1 Queue", # 0
    "Radius 2 Queue", # 1
    "Vaccinated (Timely)", # 2
    "Unvaccinated (Delayed/Missed)", # 3
    "Prophylactically Protected", # 4
    "Infected (Therapeutic Rescue)", # 5
    "Infected (Fatal)", # 6
    "Infected (Fatal, Unvaccinated)" # 7
]

# Radius 1 Flows (Efficient)
r1_source = [0, 0, 2, 2, 2, 3]
r1_target = [2, 3, 4, 5, 6, 7]
r1_value =  [80, 20, 60, 15, 5, 20]

# Radius 2 Flows (Bottleneck)
r2_source = [1, 1, 2, 2, 2, 3]
r2_target = [2, 3, 4, 5, 6, 7]
r2_value =  [40, 60, 30, 7, 3, 60]

colors = [
    'rgba(31, 119, 180, 0.8)',
    'rgba(255, 127, 14, 0.8)',
    'rgba(44, 160, 44, 0.8)',
    'rgba(214, 39, 40, 0.8)',
    'rgba(44, 160, 44, 0.8)',
    'rgba(255, 187, 120, 0.8)',
    'rgba(214, 39, 40, 0.8)',
    'rgba(150, 0, 0, 0.8)'
]

fig = go.Figure(data=[
    go.Sankey(
        node = dict(
            pad = 15,
            thickness = 20,
            line = dict(color = "black", width = 0.5),
            label = labels,
            color = colors
        ),
        link = dict(
            source = r1_source + r2_source,
            target = r1_target + r2_target,
            value = r1_value + r2_value,
            color = ['rgba(31, 119, 180, 0.3)']*len(r1_source) + ['rgba(255, 127, 14, 0.3)']*len(r2_source)
        )
    )
])

fig.update_layout(title_text="Epidemiological Trajectory: Radius 1 vs Radius 2 (Bottleneck Effect)", font_size=14)

timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
filename = f"figures/prototype_idea2_{timestamp}.png"
fig.write_image(filename, width=1200, height=800, scale=2)
print(f"Saved {filename}")
