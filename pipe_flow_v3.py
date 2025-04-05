import numpy as np
import dedalus.public as d3
import logging
import matplotlib.pyplot as plt
from mpi4py import MPI

logger = logging.getLogger(__name__)

# Parameters
Lx = 25  # Axial length (pipe diameters)
R = 1    # Pipe radius
Nx = 128 # Axial resolution
Nr = 64  # Radial resolution
Nphi = 128 # Azimuthal resolution

Re = 5300  # Reynolds number
tau = 10   # Simulation run time

# Double-cover parameters
alpha = 0.5  # Radial mapping parameter (0 < alpha < 1)

# Bases
coords = d3.CartesianCoordinates('x', 'y', 'z')
dist = d3.Distributor(coords, dtype=np.float64)
cylind = d3.CylindricalCoordinates('x', 'phi', 'r')
dealias = 3/2

# Double-cover coordinate transform
def r_transform(r):
    """Map r from [0, R] to [0, R] with double cover near origin"""
    return R * (1 - (1 - (r/R)**2)**alpha)

def r_transform_inv(r_tilde):
    """Inverse of r_transform"""
    return R * (1 - (1 - r_tilde/R)**(1/alpha))**(1/2)

# Bases with double-cover
r_basis = d3.ChebyshevT(coords['r'], size=Nr, bounds=(0, R), dealias=dealias)
phi_basis = d3.Fourier(coords['phi'], size=Nphi, bounds=(0, 2*np.pi), dealias=dealias)
x_basis = d3.Fourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)

# Fields
u = dist.VectorField(coords, name='u', bases=(x_basis, phi_basis, r_basis))
p = dist.Field(name='p', bases=(x_basis, phi_basis, r_basis))
tau_p = dist.Field(name='tau_p')
tau_u1 = dist.VectorField(coords, name='tau_u1', bases=(phi_basis, r_basis))
tau_u2 = dist.VectorField(coords, name='tau_u2', bases=(x_basis, phi_basis))

# Substitutions
nu = 1/Re  # Viscosity
div = lambda A: d3.Divergence(A, index=0)
lap = lambda A: d3.Laplacian(A, coords)
grad = lambda A: d3.Gradient(A, coords)
dot = lambda A, B: d3.DotProduct(A, B)
cross = lambda A, B: d3.CrossProduct(A, B)
integ = lambda A: d3.Integrate(d3.Integrate(d3.Integrate(A, 'x'), 'phi'), 'r')

# Problem
problem = d3.IVP([u, p, tau_p, tau_u1, tau_u2], namespace=locals())

# Momentum equation with rotation terms
problem.add_equation("dt(u) + grad(p) - nu*lap(u) + tau_u1 + tau_u2 = - dot(u, grad(u))")

# Incompressibility constraint
problem.add_equation("div(u) + tau_p = 0")

# Boundary conditions (no-slip at walls)
problem.add_equation("u(r=R) = 0")

# Regularity conditions at r=0 (handled by double-cover)
problem.add_equation("radial(u)(r=0) = 0")

# Pressure gauge condition
problem.add_equation("integ(p) = 0")

# Solver
solver = problem.build_solver(d3.RK222)
solver.stop_sim_time = tau

# Initial conditions (parabolic profile + noise)
u['g'][0] = 2 * (1 - (dist.local_grid(cylind['r'])/R)**2)
noise = dist.VectorField(coords, bases=(x_basis, phi_basis, r_basis))
noise.fill_random('g', seed=42)
noise.low_pass_filter(scales=0.25)
u['g'] += 0.01 * noise['g']

# Flow properties for monitoring
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(d3.dot(u, u), name='u2')

# CFL
CFL = d3.CFL(solver, initial_dt=1e-3, cadence=10, safety=0.3, threshold=0.1)
CFL.add_velocity(u)

# Main loop
try:
    logger.info('Starting main loop')
    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)
        if (solver.iteration-1) % 100 == 0:
            logger.info(f"Iteration={solver.iteration}, Time={solver.sim_time:.2f}, "
                       f"dt={timestep:.2e}, KE={flow.volume_integral('u2'):.2e}")
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()

# Post-processing
# (Add your analysis code here - mean profiles, spectra, etc.)