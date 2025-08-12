import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)

#### Parameters ###
Lx, Ly = (0.6*np.pi, 2.0)
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
initial_dt=1e-2
# Create bases and domain
nx, ny = 48, 64 #54, 129, 42
#nx, ny, nz = 192, 129, 160 # larger box. Kim Moin and Moser 

coords = d3.CartesianCoordinates('x', 'y')
dist = d3.Distributor(coords, dtype=np.float64)
xbasis = d3.RealFourier(coords['x'], size=nx, bounds=(0, Lx), dealias=3/2)
ybasis = d3.Chebyshev(coords['y'], size=ny, bounds=(-Ly/2, Ly/2), dealias=3/2)

# Fields
T = dist.Field(name='T', bases=(xbasis,ybasis))
tau_T1 = dist.Field(name='tau_T1', bases=(xbasis))
tau_T2 = dist.Field(name='tau_T2', bases=(xbasis))


# Substitutions
#dPdx = -Retau**2/Re**2
dPdx = -2/Re
x, y = dist.local_grids(xbasis, ybasis)
ex, ey = coords.unit_vector_fields(dist)
lift_basis = ybasis.derivative_basis(1) # Chebyshev U basis
lift = lambda A: d3.Lift(A, lift_basis, -1) # Shortcut for multiplying by U_{N-1}(y)
grad_T = d3.grad(T) - ey*lift(tau_T1) # First-order reduction
x_average = lambda A: d3.Average(A,'x')
xz_average = lambda A: d3.Average(A, 'x')
vol_average = lambda A: d3.Average(d3.Average(A, 'x'),'y')
dx = lambda A: d3.Differentiate(A,coords['x'])
sin = lambda A: np.sin(A)
# Problem


U = dist.Field(name='U',bases=(ybasis))
U['g'] = 1-y**2

problem = d3.IVP([T, tau_T1, tau_T2], namespace=locals())
problem.namespace.update({'t':problem.time})
problem.add_equation("dt(T) - 1/(Re*Pr)*div(grad_T) + lift(tau_T2) + U*dx(T)= 0")
problem.add_equation("T(y=+1)=0")
problem.add_equation("T(y=-1)=A0+A*sin(omega*t)")

# Build Solver
dt = 0.002 # 0.001
stop_sim_time = 100
fh_mode = 'overwrite'
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time




snapshots = solver.evaluator.add_file_handler('snapshots_channel', sim_dt=20, max_writes=600)

snapshots.add_task(T, name='temperature')

snapshots_thermal = solver.evaluator.add_file_handler('snapshots_channel_thermal',sim_dt=0.1,max_writes=1000)
snapshots_thermal.add_task(xz_average(d3.grad(T)@ey),name = 'dTdy')
snapshots_thermal.add_task(xz_average(T),name = 'T')

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=20) # changed cadence from 10 to 50

flow.add_property(T, name='T')
flow.add_property(d3.grad(T)@ey, name='dTdy')


# Main loop
startup_iter = 10
try:
    logger.info('Starting main loop')
    while solver.proceed:
        timestep = initial_dt
        solver.step(timestep)
        if (solver.iteration-1) % 10 == 0:
            T = flow.max('T')
            dTdy = flow.max('dTdy')
            logger.info('Iteration=%i, Time=%e, dt=%e, max(T)=%f, max(dTdy)=%f' %(solver.iteration, solver.sim_time, timestep, T, dTdy))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()
