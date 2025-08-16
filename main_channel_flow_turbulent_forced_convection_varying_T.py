import numpy as np
import dedalus.public as d3
import logging
#from dedalus.core.operators import interp

logger = logging.getLogger(__name__)

#### Parameters ###

#Re = 16200 # U_b*H/nu
#Re_tau=180
Re_tau=550

Pr=0.71 #Prandtl number

### domain size
#Lx, Ly, Lz = (0.6*np.pi, 2.0, 0.18*np.pi) #Re_tau=180, minimal box
#Lx, Ly, Lz = (4.0*np.pi, 2.0, 2.0*np.pi) #Re_tau=180, regular box
Lx, Ly, Lz = (2.0*np.pi, 2.0, np.pi) #Re_tau=550, Hoyas's thermal box

### resolutions
#nx, ny, nz = 48, 64, 42 #54, 129, 42
#nx, ny, nz = 192, 129, 160 #Re_tau=180, Kim Moin and Moser resolution. 
nx, ny, nz = 288, 180, 240 #Re_tau =550, Lx=2pi, Lz=pi Hoyas box. 

dtype = np.float64
#stop_sim_time = 50
timestepper = d3.RK222
max_timestep = 0.1  # 0.125 to 0.1

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
dPdx = -1
x, y, z = dist.local_grids(xbasis, ybasis, zbasis)
ex, ey, ez = coords.unit_vector_fields(dist)
lift_basis = ybasis.derivative_basis(1) # Chebyshev U basis
lift = lambda A: d3.Lift(A, lift_basis, -1) # Shortcut for multiplying by U_{N-1}(y)
grad_u = d3.grad(u) - ey*lift(tau_u1) # Operator representing G
grad_T = d3.grad(T) - ey*lift(tau_T1) # First-order reduction
x_average = lambda A: d3.Average(A,'x')
xz_average = lambda A: d3.Average(d3.Average(A, 'x'), 'z')
vol_average = lambda A: d3.Average(d3.Average(d3.Average(A, 'x'), 'z'),'y')

#sin = lambda A: np.sin(A)
# Problem

problem = d3.IVP([p, u, T, tau_p, tau_u1, tau_u2, tau_T1, tau_T2], namespace=locals())
problem.namespace.update({'t':problem.time})
problem.add_equation("trace(grad_u) + tau_p = 0")
problem.add_equation("dt(u) - 1/Re_tau*div(grad_u) + grad(p) + lift(tau_u2) =-dPdx*ex -dot(u,grad(u))")
problem.add_equation("dt(T) - 1/(Re_tau*Pr)*div(grad_T) + lift(tau_T2) = - u@grad(T) + (u@ex)/vol_average(u@ex)")
problem.add_equation("u(y=-1) = 0") # change from -1 to -0.5
problem.add_equation("u(y=+1) = 0") #change from 1 to 0.5
problem.add_equation("integ(p) = 0")
#problem.add_equation("T(y=+1)=A0+A*sin(omega*t)")
problem.add_equation("T(y=+1)=0")
problem.add_equation("T(y=-1)=0")

# Build Solver
dt = 0.002 # 0.001
stop_sim_time = 100
fh_mode = 'overwrite'
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

np.random.seed(0)
#u['g'][0] = (1-y**2) 

kappa  = 0.426
A      = 25.4
nu     = 1.0 

nu_T = lambda eta: (nu/2)*(1+(kappa**2*Re_tau**2/9.0)*(1-eta**2)**2*(1+2*eta**2)**2*(1-np.exp((np.abs(eta)-1)*Re_tau/A))**2)**0.5+nu/2

dUdy = lambda eta: -Re_tau*eta/nu_T(eta)

from scipy.integrate import quad

U_plus = np.zeros_like(y)
#print(y)
#print(len(y[0]))
for j in range(0, len(y[0])):
    #print(j)
    #print('nu_T(y)',nu_T(y[0][j]))
    #print('dUdy(y)',dUdy(y[0][j]))
    result, error =quad(dUdy, -1, y[0][j])
    U_plus[0][j] = result
    print('U(y)=',U_plus[0][j],'at y=',y[0][j])
    

u['g'][0]=U_plus[np.newaxis, :, np.newaxis]+np.random.randn(*u['g'][0].shape) * 1e-6*np.sin(np.pi*(y+1)*0.5)

#This is random noise to trigger transition to turbulence
#+ np.random.randn(*u['g'][0].shape) * 1e-6*np.sin(np.pi*(y+1)*0.5) # Laminar solution (plane Poiseuille)+  random perturbation

#Full 3D snapshots, every sim_dt=20
snapshots = solver.evaluator.add_file_handler('snapshots_channel', sim_dt=20, max_writes=200)
snapshots.add_task(u, name='velocity')
snapshots.add_task(T, name='temperature')

#2D slicing from the 3D data, every sim_dt=1
snapshots_2D = solver.evaluator.add_file_handler('snapshots_channel_2D',sim_dt=1,max_writes=4000)
snapshots_2D.add_task(u(x=0), name='u_yz')
snapshots_2D.add_task(u(z=0), name='u_xy')
snapshots_2D.add_task(u(y=0), name='u_xz_mid')
snapshots_2D.add_task(u(y=5/Re_tau), name='u_xz_viscous')
snapshots_2D.add_task(u(y=15/Re_tau), name='u_xz_buffer')
snapshots_2D.add_task(u(y=50/Re_tau), name='u_xz_log')

snapshots_2D.add_task(T(x=0), name='T_yz')
snapshots_2D.add_task(T(z=0), name='T_xy')
snapshots_2D.add_task(T(y=0), name='T_xz_mid')
snapshots_2D.add_task(T(y=5/Re_tau), name='T_xz_viscous')
snapshots_2D.add_task(T(y=15/Re_tau), name='T_xz_buffer')
snapshots_2D.add_task(T(y=50/Re_tau), name='T_xz_log')

#1D statistics, every sim_dt=0.1
snapshots_stress = solver.evaluator.add_file_handler('snapshots_channel_stress', sim_dt=0.1, max_writes=40000)
snapshots_stress.add_task(xz_average(u)@ex,name = 'ubar')
snapshots_stress.add_task(d3.grad(xz_average(u)@ex)@ey,name = 'dudy')
snapshots_stress.add_task(vol_average(u@ex),name = 'u_bulk')
snapshots_stress.add_task(xz_average(((u-xz_average(u))@ex)**2),name = 'u_prime_u_prime')
snapshots_stress.add_task(xz_average(((u-xz_average(u))@ey)**2),name = 'v_prime_v_prime')
snapshots_stress.add_task(xz_average(((u-xz_average(u))@ez)**2),name = 'w_prime_w_prime')
snapshots_stress.add_task(xz_average(((u-xz_average(u))@ex)*(u-xz_average(u))@ey),name = 'u_prime_v_prime')

snapshots_stress.add_task(xz_average(T),name = 'T')
snapshots_stress.add_task(xz_average(d3.grad(T)@ey),name = 'dTdy')
snapshots_stress.add_task(vol_average((u@ex)*T)/vol_average(u@ex),name = 'T_bulk')
snapshots_stress.add_task(xz_average((T-xz_average(T))**2),name = 'T_prime_T_prime')
snapshots_stress.add_task(xz_average(((u-xz_average(u))@ex)*(T-xz_average(T))),name = 'u_prime_T_prime')
snapshots_stress.add_task(xz_average(((u-xz_average(u))@ey)*(T-xz_average(T))),name = 'v_prime_T_prime')
snapshots_stress.add_task(xz_average(((u-xz_average(u))@ez)*(T-xz_average(T))),name = 'w_prime_T_prime')

#snapshots_thermal = solver.evaluator.add_file_handler('snapshots_channel_thermal',sim_dt=0.1,max_writes=40000)


# CFL
CFL = d3.CFL(solver, initial_dt=dt, cadence=5, safety=0.5, threshold=0.05,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u) # changed threshold from 0.05 to 0.01

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=20) # changed cadence from 10 to 50
flow.add_property(np.sqrt((u-xz_average(u))@(u-xz_average(u)))/2, name='TKE')

flow.add_property(xz_average(T), name='T_bar')
flow.add_property(xz_average(u@ex), name='u_bar')


# Main loop
startup_iter = 10
try:
    logger.info('Starting main loop')
    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)
        if (solver.iteration-1) % 10 == 0:
            TKE_max = flow.max('TKE')
            T_bar_max = flow.max('T_bar')
            u_bar_max = flow.max('u_bar')

            logger.info('Iteration=%i, Time=%e, dt=%e, max(TKE)=%f, max(u_bar)=%f, max(T_bar)=%f' %(solver.iteration, solver.sim_time, timestep, TKE_max, u_bar_max, T_bar_max))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()
