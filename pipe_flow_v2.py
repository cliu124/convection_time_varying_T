import numpy as np
from dedalus import public as de
import logging
logger = logging.getLogger(__name__)

# Parameters
R = 1.0            # Pipe radius
Lz = 2*np.pi       # Length in z
Ntheta = 64
Nr = 64
Nz = 64
nu = 0.01           # Kinematic viscosity

# Bases
r_basis = de.Chebyshev('r', Nr, interval=(-R, R), dealias=3/2)
theta_basis = de.Fourier('theta', Ntheta, interval=(0, 2*np.pi), dealias=3/2)
z_basis = de.Fourier('z', Nz, interval=(0, Lz), dealias=3/2)

domain = de.Domain([r_basis, theta_basis, z_basis], grid_dtype=np.float64)

# Variables (primes are auxiliary gradients)
problem = de.IVP(domain, variables=[
    'p', 'u', 'v', 'w',
    'ur', 'vr', 'wr',
    'utheta', 'vtheta', 'wtheta',
    'uz', 'vz', 'wz'
])

# Parities (r, theta, z)
parities = {
    'u': (-1, 1, 1),   'v': (1, -1, 1),   'w': (1, 1, -1),   'p': (1, 1, 1),
    'ur': (-1, 1, 1),  'vr': (1, -1, 1),  'wr': (1, 1, -1),
    'utheta': (-1, -1, 1), 'vtheta': (1, 1, 1), 'wtheta': (1, -1, -1),
    'uz': (-1, 1, 1),  'vz': (1, -1, 1),  'wz': (1, 1, -1),
}

for name, parity in parities.items():
    problem.meta[name]['r', 'theta', 'z'] = parity

# Operators
r = domain.grid(0)
theta = domain.grid(1)
z = domain.grid(2)

# Equations

# Continuity
problem.parameters['nu']=nu
problem.add_equation("ur + u/r + utheta/r + vz + wr = 0")

# Momentum equations
problem.add_equation("dt(u) + dr(p) - nu*(dr(ur) + ur/r + utheta/r**2 + d(dz, uz)) = - (u*ur + v*utheta/r + w*uz - v**2/r)")
problem.add_equation("dt(v) + (1/r)*dtheta(p) - nu*(dr(vr) + vr/r + vtheta/r**2 + d(dz, vz) - 2*utheta/r**2) = - (u*vr + v*vtheta/r + w*vz + u*v/r)")
problem.add_equation("dt(w) + dz(p) - nu*(dr(wr) + wr/r + wtheta/r**2 + d(dz, wz)) = - (u*wr + v*wtheta/r + w*wz)")


# Auxiliary derivatives
problem.add_equation("ur - dr(u) = 0")
problem.add_equation("vr - dr(v) = 0")
problem.add_equation("wr - dr(w) = 0")

problem.add_equation("utheta - dtheta(u) = 0")  # replace if angular derivatives are wanted
problem.add_equation("vtheta - dtheta(v) = 0")  # placeholder for ∂θ
problem.add_equation("wtheta - dtheta(w) = 0")

problem.add_equation("uz - dz(u) = 0")
problem.add_equation("vz - dz(v) = 0")
problem.add_equation("wz - dz(w) = 0")

# Solver
solver = problem.build_solver(de.timesteppers.RK443)
logger.info("Solver built")

# Initial conditions
u = solver.state['u']
v = solver.state['v']
w = solver.state['w']

# Grid
r = domain.grid(0)
theta = domain.grid(1)
z = domain.grid(2)

# Initial velocity: parabolic profile in axial direction
w['g'] = (R**2 - r**2) * np.cos(z)
u['g'] = r * np.sin(theta) * np.sin(z)
v['g'] = r * np.cos(theta) * np.sin(z)

# Time stepping
dt = 1e-3
stop_sim_time = 0.2
solver.stop_sim_time = stop_sim_time

# Run simulation
while solver.ok:
    solver.step(dt)
    if solver.iteration % 10 == 0:
        print(f"Iteration {solver.iteration}, Time = {solver.sim_time:.4f}")

print("Navier-Stokes simulation complete.")
