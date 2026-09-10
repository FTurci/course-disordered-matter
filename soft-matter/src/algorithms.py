import numpy as np

def ls_packing(N, L, sigma0=0.1, growth_rate=0.01, dt=0.01, steps=10000):
    pos = np.random.uniform(0, L, (N, 3))
    vel = np.random.normal(0, 1, (N, 3))
    vel /= np.linalg.norm(vel, axis=1)[:, None]
    diam = np.full(N, sigma0)

    def minimum_image(rij):
        return rij - L * np.round(rij / L)

    for step in range(steps):
        pos += vel * dt
        pos %= L
        diam += growth_rate * dt

        for i in range(N):
            for j in range(i+1, N):
                rij = minimum_image(pos[i] - pos[j])
                dist = np.linalg.norm(rij)
                if dist < 0.5 * (diam[i] + diam[j]):
                    # Simple elastic collision
                    vij = vel[i] - vel[j]
                    n = rij / dist
                    dv = np.dot(vij, n)
                    if dv < 0:
                        impulse = n * dv
                        vel[i] -= impulse
                        vel[j] += impulse
    return pos, diam
