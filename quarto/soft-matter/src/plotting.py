import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['xtick.direction'] = "in"
plt.rcParams['ytick.direction'] = "in"
plt.rcParams['legend.frameon'] = False
import inspect



def simple_plot(arr1, arr2):
    
    # Get caller local variables for name lookup
    fig = plt.figure(figsize=(5, 2))

    caller_locals = inspect.currentframe().f_back.f_locals
    # Find names by matching objects
    arr1_name = next((name for name, val in caller_locals.items() if val is arr1), 'arr1')
    arr2_name = next((name for name, val in caller_locals.items() if val is arr2), 'arr2')
    plt.plot(arr1,arr2)
    
    if "_"  in arr1_name:   
        arr1_name = arr1_name.replace("_", r"_{")
        arr1_name+=r"}"

    if "_"  in arr2_name:   
        arr2_name = arr2_name.replace("_", r"_{")
        arr2_name+=r"}"
        
    plt.xlabel(rf"${arr1_name}$")
    plt.ylabel(rf"${arr2_name}$")
    plt.tight_layout()
    plt.show()



def multi_plot(*args, xlabel="x", ylabel="y",zero='true'):
    # args[0] is x data
    x = args[0]

    # Get caller local variables for name lookup
    caller_locals = inspect.currentframe().f_back.f_locals

    fig = plt.figure(figsize=(5.1, 3.5))

    # Loop over all y arrays passed after x
    for a in args[1:]:
        # Find the first variable name in caller locals whose value is this object
        arr_name = next((name for name, val in caller_locals.items() if val is a), None)
        if arr_name is None:
            arr_name = 'data'

        # Replace underscores with LaTeX subscript
        if "_" in arr_name:
            arr_name = arr_name.replace("_", r"_{") + r"}"

        plt.plot(x, a, label=fr"${arr_name}$", alpha=0.7)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(x,np.zeros_like(x),':k')
    plt.legend()
    plt.tight_layout()
    plt.show()

import plotly.graph_objects as go

# Plotly 3D line plot
fig = go.Figure(data=[go.Scatter3d(
    x=x, y=y, z=z,
    mode='lines+markers',
    marker=dict(size=3, color=np.arange(N+1), colorscale='Viridis'),
    line=dict(color='blue', width=4)
)])
fig.update_layout(
    title="3D Random Walk (Freely-Jointed Chain)",
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z'
    ),
    margin=dict(l=0, r=0, b=0, t=30)
)
fig.show()
