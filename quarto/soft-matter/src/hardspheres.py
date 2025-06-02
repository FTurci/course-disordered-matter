import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Line3DCollection

import numba

def generate_fcc_lattice(n_cells, lattice_const):
    basis = np.array([[0,0,0],[0.5,0.5,0],[0.5,0,0.5],[0,0.5,0.5]])
    pos = []
    for x in range(n_cells):
        for y in range(n_cells):
            for z in range(n_cells):
                origin = np.array([x,y,z]) * lattice_const
                for b in basis:
                    pos.append(origin + b * lattice_const)
    return np.array(pos)

def sphere(center, radius, resolution=10):
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
    return x, y, z

@numba.njit
def has_overlap(positions, radius):
    N = len(positions)
    for i in range(N):
        for j in range(i+1, N):
            delta = positions[i] - positions[j]
            delta -= np.round(delta / box_length) * box_length  # PBC
            dist = np.linalg.norm(delta)
            if dist < 2 * radius:
                return True
    return False

@numba.njit
def monte_carlo_step(positions, radius, delta=0.1, fixed_idx=0):

    N = len(positions)
    idx = np.random.randint(N)
    if idx == fixed_idx:
        return  # skip moving the fixed particle
    move = (2 * np.random.rand(3) - 1) * delta

    old_pos = positions[idx].copy()
    positions[idx] = (positions[idx] + move) % box_length

    if has_overlap(positions, radius):
        positions[idx] = old_pos

def draw_box(ax, box_length):
    r = [0, box_length]
    points = np.array([[x,y,z] for x in r for y in r for z in r])
    # Edges between cube corners
    edges = [
        [points[0], points[1]], [points[0], points[2]], [points[0], points[4]],
        [points[1], points[3]], [points[1], points[5]],
        [points[2], points[3]], [points[2], points[6]],
        [points[3], points[7]],
        [points[4], points[5]], [points[4], points[6]],
        [points[5], points[7]],
        [points[6], points[7]]
    ]
    lc = Line3DCollection(edges, colors='black', linewidths=1, linestyles='dashed', alpha=0.7)
    ax.add_collection(lc)


class Plotter3D:
    def __init__(self, positions, radius):
        self.positions = positions
        self.radius = radius
        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_box_aspect([1, 1, 1])
        self.ax.set_xlim(0, box_length)
        self.ax.set_ylim(0, box_length)
        self.ax.set_zlim(0, box_length)
        self.ax.axis('off')

        draw_box(self.ax, box_length)  # Draw the box once
        self.resolution = 10
        self.alpha = 1.0
        self.color = 'orange'
        self.sphere_surfaces = []

        for pos in self.positions:
            x, y, z = sphere(pos, radius, resolution=self.resolution)
            surf = self.ax.plot_surface(x, y, z, color=self.color, alpha=self.alpha, linewidth=0)
            self.sphere_surfaces.append(surf)

    def create_animation(self,html=True):
        def update(frame):
            for _ in range(self.positions.shape[0]*20):
                monte_carlo_step(self.positions, radius, delta=0.1)

            for surf in self.sphere_surfaces:
                surf.remove()
            self.sphere_surfaces.clear()

            for pos in positions:
                x, y, z = sphere(pos, radius, resolution=self.resolution)
                self.sphere_surfaces.append(self.ax.plot_surface(x, y, z,  color=self.color, alpha=self.alpha, linewidth=0))

            return self.sphere_surfaces

        ani = FuncAnimation(self.fig, update, frames=200, interval=100, blit=False)
        if html == True:
            from IPython.display import HTML
            return HTML(ani.to_jshtml())
        else:   
            return ani
        
    def create_trajectory(self,filename="output.xyz", n_steps=1000, delta=0.1):
        import tqdm
        with open(filename, 'w') as f:
          
            for _ in tqdm.tqdm(range(n_steps)): 
                for p in range(self.positions.shape[0]):
                    monte_carlo_step(self.positions, self.radius, delta=delta)
                f.write(f"{positions.shape[0]}\nAtoms\n")
                for pos in self.positions:    
                    f.write(f"A {pos[0]/(2*self.radius)} {pos[1]/(2*self.radius)} {pos[2]/(2*self.radius)}\n")
    
n_cells = 2
box_length = n_cells
N = 4 * n_cells**3
packing_fraction = 0.4922

vol_box = box_length ** 3
vol_spheres_total = packing_fraction * vol_box
vol_sphere = vol_spheres_total / N
radius = (3 * vol_sphere / (4 * np.pi)) ** (1/3)

positions = generate_fcc_lattice(n_cells, lattice_const=box_length / n_cells)

plotter = Plotter3D(positions, radius)
# ani = plotter.create_animation(html=False)
# plt.show(
plotter.create_trajectory(filename="output.xyz", n_steps=10000, delta=0.1)