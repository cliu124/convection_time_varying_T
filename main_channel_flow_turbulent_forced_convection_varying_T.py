import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)

#### Parameters ###
Lx, Ly, Lz = (0.6*np.pi, 2.0, 0.18*np.pi)
#Lx, Ly, Lz = (4.0*np.pi, 2.0, 2.0*np.pi)

#Re = 16200 # U_b*H/nu
Re=350
Pr=1

A0=1
A=0.5
omega=0.1

#Retau = 180 # = u_tau*H/nu
dtype = np.float64
#stop_sim_time = 50
timestepper = d3.RK222
max_timestep = 0.1  # 0.125 to 0.1

# Create bases and domain
nx, ny, nz = 48, 64, 42 #54, 129, 42
#nx, ny, nz = 192, 129, 160 # larger box. Kim Moin and Moser 

coords = d3.CartesianCoordinates('x', 'y','z')
dist = d3.Distributor(coords, dtype=np.float64)
xbasis = d3.RealFourier(coords['x'], size=nx, bounds=(0, Lx), dealias=3/2)
zbasis = d3.RealFourier(coords['z'], size=nz, bounds=(0, Lz), dealias=3/2)
ybasis = d3.Chebyshev(coords['y'], size=ny, bounds=(-Ly/2, Ly/2), dealias=3/2)

# Fields
p = dist.Field(name='p', bases=(xbasis,ybasis,zbasis))
T = dist.Field(name='T', bases=(xbasis,ybasis,zbasis))
u = dist.VectorField(coords, name='u', bases=(xbasis,ybasis,zbasis))
tau_u1 = dist.VectorField(coords, name='tau_u1', bases=(xbasis,zbasis))
tau_u2 = dist.VectorField(coords, name='tau_u2', bases=(xbasis,zbasis))
tau_T1 = dist.Field(name='tau_T1', bases=(xbasis,zbasis))
tau_T2 = dist.Field(name='tau_T2', bases=(xbasis,zbasis))
tau_p = dist.Field(name='tau_p')

# Substitutions
#dPdx = -Retau**2/Re**2
dPdx = -2/Re
x, y, z = dist.local_grids(xbasis, ybasis, zbasis)
ex, ey, ez = coords.unit_vector_fields(dist)
lift_basis = ybasis.derivative_basis(1) # Chebyshev U basis
lift = lambda A: d3.Lift(A, lift_basis, -1) # Shortcut for multiplying by U_{N-1}(y)
grad_u = d3.grad(u) - ey*lift(tau_u1) # Operator representing G
grad_T = d3.grad(T) - ey*lift(tau_T1) # First-order reduction
x_average = lambda A: d3.Average(A,'x')
xz_average = lambda A: d3.Average(d3.Average(A, 'x'), 'z')
vol_average = lambda A: d3.Average(d3.Average(d3.Average(A, 'x'), 'z'),'y')

sin = lambda A: np.sin(A)
# Problem

problem = d3.IVP([p, u, T, tau_p, tau_u1, tau_u2, tau_T1, tau_T2], namespace=locals())
problem.namespace.update({'t':problem.time})
problem.add_equation("trace(grad_u) + tau_p = 0")
problem.add_equation("dt(u) - 1/Re*div(grad_u) + grad(p) + lift(tau_u2) =-dPdx*ex -dot(u,grad(u))")
problem.add_equation("dt(T) - 1/(Re*Pr)*div(grad_T) + lift(tau_T2) = - u@grad(T)")
problem.add_equation("u(y=-1) = 0") # change from -1 to -0.5
problem.add_equation("u(y=+1) = 0") #change from 1 to 0.5
problem.add_equation("integ(p) = 0")
#problem.add_equation("T(y=+1)=A0+A*sin(omega*t)")
problem.add_equation("T(y=+1)=0")
problem.add_equation("T(y=-1)=A0+A*sin(omega*t)")

# Build Solver
dt = 0.002 # 0.001
stop_sim_time = 100
fh_mode = 'overwrite'
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

np.random.seed(0)
u['g'][0] = (1-y**2) 

#This is random noise to trigger transition to turbulence
#+ np.random.randn(*u['g'][0].shape) * 1e-6*np.sin(np.pi*(y+1)*0.5) # Laminar solution (plane Poiseuille)+  random perturbation


snapshots = solver.evaluator.add_file_handler('snapshots_channel', sim_dt=20, max_writes=600)

snapshots.add_task(u, name='velocity')
snapshots.add_task(T, name='temperature')

snapshots_stress = solver.evaluator.add_file_handler('snapshots_channel_stress', sim_dt=5, max_writes=400)
snapshots_stress.add_task(xz_average(u),name = 'ubar')
snapshots_stress.add_task(xz_average(((u-xz_average(u))@ex)**2),name = 'u_prime_u_prime')
snapshots_stress.add_task(xz_average(((u-xz_average(u))@ey)**2),name = 'v_prime_v_prime')
snapshots_stress.add_task(xz_average(((u-xz_average(u))@ez)**2),name = 'w_prime_w_prime')
snapshots_stress.add_task(xz_average(((u-xz_average(u))@ex)*(u-xz_average(u))@ey),name = 'u_prime_v_prime')

snapshots_thermal = solver.evaluator.add_file_handler('snapshots_channel_thermal',sim_dt=0.1,max_writes=1000)
snapshots_thermal.add_task(xz_average(d3.grad(T)@ey),name = 'dTdy')
snapshots_thermal.add_task(xz_average(T),name = 'T')

# CFL
CFL = d3.CFL(solver, initial_dt=dt, cadence=5, safety=0.5, threshold=0.05,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u) # changed threshold from 0.05 to 0.01

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=20) # changed cadence from 10 to 50
flow.add_property(np.sqrt(u@u)/2, name='TKE')

flow.add_property(T, name='T')
flow.add_property(d3.grad(T)@ey, name='dTdy')


# Main loop
startup_iter = 10
try:
    logger.info('Starting main loop')
    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)
        if (solver.iteration-1) % 10 == 0:
            max_TKE = flow.max('TKE')
            T = flow.max('T')
            dTdy = flow.max('dTdy')
            logger.info('Iteration=%i, Time=%e, dt=%e, max(TKE)=%f, max(T)=%f, max(dTdy)=%f' %(solver.iteration, solver.sim_time, timestep, max_TKE, T, dTdy))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()
