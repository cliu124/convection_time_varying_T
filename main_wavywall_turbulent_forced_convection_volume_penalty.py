import numpy as np
import dedalus.public as d3
import logging
import matplotlib.pyplot as plt
from mpi4py import MPI
from scipy.special import erf
from scipy.optimize import minimize_scalar
import matplotlib.colors as mcolors
import time
logger = logging.getLogger(__name__)

#### Parameters ###

#Re = 16200 # U_b*H/nu
Re_tau=180
#Re_tau=550

Pr=0.71 #Prandtl number

A=0.1 #[0.1,0.2,0.3,0.4] pressure gradient oscillation amplitude
omega=0.2 #[0.2,0.5,1,2,5] pressure gradient oscillation frequency
epsilon = 2*0.27 # 0.125
mask_option = "opt_erf" 

### domain size
#Lx, Ly, Lz = (0.6*np.pi, 2.0, 0.18*np.pi) #Re_tau=180, minimal box
#Lx, Ly, Lz = (4.0*np.pi, 2.0, 2.0*np.pi) #Re_tau=180, regular box
Lx, Ly, Lz = (4.0, 2.0 + 2*epsilon, 0.05*np.pi)# Validation Case Kim Joo Choo paper Lx ~ 3.0

#Lx, Ly, Lz = (2.0*np.pi, 2.0, np.pi) #Re_tau=550, Hoyas's thermal box

### resolutions
#nx, ny, nz = 48, 64, 42 #54, 129, 42
#nx, ny, nz = 192, 129, 160 #Re_tau=180, Kim Moin and Moser resolution. 
nx, ny, nz = 218, 320, 4 #Re_tau=180, double the vertical resolution

#nx, ny, nz = 288, 512, 240 #Re_tau =550, Lx=2pi, Lz=pi Hoyas box. 
#nx, ny, nz = 256, 416, 240 #Re_tau=550, parallel in y direction. Fourier direction


restart=0
checkpoint_path='/mnt/c/Users/jinog/Documents/dedalus/channelflow/snapshots_channel/snapshots_channel_s1/snapshots_channel_s1.h5'
load_time= 99

dtype = np.float64
#stop_sim_time = 50
timestepper = d3.RK443
max_timestep = 0.05  # 0.125 to 0.1

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
inv_k = dist.Field(name='inv_k', bases=(xbasis,ybasis,zbasis)) # stiffness function
mask = dist.Field(name='mask', bases=(xbasis, ybasis, zbasis)) # mask function

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
sin = lambda A: np.sin(A)


def build_vp_mask():
    # convert x, y and z into 1 dimension vectors 
    x_flat = x.flatten()
    y_flat = y.flatten()
    z_flat = z.flatten()

    nx_global = xbasis.shape[0] # Global number of points or indices in the x-direction, not local
    ny_global = ybasis.shape[0] # Global number of points or indices in the y-direction, not local
    nz_global = zbasis.shape[0] # Global number of points or indices in the z-direction, not local

    sin = lambda A: np.sin(A)
    # Problem
    ### Computing  ∂u/∂x , ∂u/∂y , ∂v/∂x , ∂v/∂y ------------------------------------###
    ### -----------------------------------------------------------------------------###
    dx= lambda A: d3.Differentiate(A,coords['x'])
    dy= lambda A: d3.Differentiate(A,coords['y'])

    #ux = u @ ex  # projection of u onto x-direction
    #uy = u @ ey  # projection of u onto y-direction
    #uz = u @ ez  # projection of u onto z-direction

    dudx = dx(u @ ex)   # ∂u/∂x
    dudy = dy(u @ ex)   # ∂u/∂y
    dvdx = dx(u @ ey)   # ∂v/∂x
    dvdy = dy(u @ ey)   # ∂v/∂y 

    ### -----------------------------------------------------------------------------###
    ### -----------------------------------------------------------------------------### 

    y0 = 1.0  #changed from 0.6 to 1.0 to 0.8 for validation
    A1 = epsilon  # Amplitude for first wave
    A2 = epsilon  # Amplitude for second wave (negative to flip direction) 
    y_sin1 = -y0 - A1 * np.sin(2 * np.pi / Lx * x)  # First wave
    y_sin2 =  y0 + A2 * np.sin(2 * np.pi / Lx * x)  # Second wave 
    dy = (Ly + epsilon)/ny_global
    mask_threshold = 4*0.5*dy  # Adjust as needed, change from 0.5dy to 6*0.5 * dy for the last case

    def signed_distance_to_wavy_walls_sampled(x_pt, y_pt, y_base1, A1, y_base2, A2, Lx, nsample):
        s_vals = np.linspace(0, Lx, nsample)

        # Evaluate wall y-values
        y1_vals = y_base1 + A1 * np.sin(2 * np.pi * s_vals / Lx)  # bottom wall
        y2_vals = y_base2 + A2 * np.sin(2 * np.pi * s_vals / Lx)  # top wall

        # Distance squared to each wall
        dist2_1 = (x_pt - s_vals)**2 + (y_pt - y1_vals)**2
        dist2_2 = (x_pt - s_vals)**2 + (y_pt - y2_vals)**2

        min_dist1 = np.min(dist2_1)
        min_dist2 = np.min(dist2_2)

        d1 = np.sqrt(min_dist1)
        d2 = np.sqrt(min_dist2)

        # Get closest point on wall to determine sign
        s1_closest = s_vals[np.argmin(dist2_1)]
        s2_closest = s_vals[np.argmin(dist2_2)]

        y1_closest = y_base1 + A1 * np.sin(2 * np.pi * s1_closest / Lx)
        y2_closest = y_base2 + A2 * np.sin(2 * np.pi * s2_closest / Lx)


        if d1 < d2:
            sign = 1 if y_pt > y1_closest else -1  # you are above the bottom wall â†’ fluid
            return sign * d1
        else:
            sign = 1 if y_pt < y2_closest else -1  # you are below the top wall â†’ fluid
            return sign * d2    

    y_local = np.squeeze(y[0, :, 0])       # shape (ny_local,)

    d_perp = np.zeros((len(y_local), nx))  # shape = (ny_local, nx)

    for j in range(len(y_local)):
        for i in range(nx):
            x_pt = x[i]  # Only need x/y from mid-z plane
            y_pt = y_local[j]
            d_perp[j, i] = signed_distance_to_wavy_walls_sampled(x_pt, y_pt, -y0, -A1, y0, A2, Lx, nsample=6000)


    # Define binary region identifiers (not normalized values)
    solid_region = d_perp < -mask_threshold
    fluid_region = d_perp > mask_threshold
    interface_region = ~np.logical_or(solid_region, fluid_region)

    # Optional visualization (triple mask values: 1 for solid, 0.5 for interface, 0 for fluid)
    binary_mask = np.zeros_like(d_perp)
    binary_mask[solid_region] = 1
    binary_mask[interface_region] = 0.5
    binary_mask[fluid_region] = 0


    print("Binary mask computed successfully.")

    comm = MPI.COMM_WORLD
    rank = comm.rank

    # --- Gather mask data on rank 0 ---
    gathered_mask = comm.gather(binary_mask, root=0)

    # These are in shape (nx_local, ny_local), so we transpose them
    gathered_x = comm.gather(np.transpose(x[:, :, 0]), root=0)  # shape (ny_local, nx)
    gathered_y = comm.gather(np.transpose(y[:, :, 0]), root=0)  # shape (ny_local, nx)

    if rank == 0:
        full_mask = np.concatenate(gathered_mask, axis=0)
        full_y = np.concatenate(gathered_y, axis=0)
        full_x = gathered_x[0]
        full_x, full_y = np.meshgrid(full_x.flatten(), full_y.flatten())

        print("full_mask shape:", full_mask.shape)
        print("full_x shape:", full_x.shape)
        print("full_y shape:", full_y.shape)

        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        plt.pcolormesh(full_x, full_y, full_mask, shading='auto', cmap='gray')
        plt.colorbar(label="Mask Value")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Binary Mask for Symmetric Wavy Walls (MPI)")
        plt.tight_layout()
        plt.grid(False)
        plt.savefig("binary_mask_preview.png", dpi=150)
        print("Saved plot to binary_mask_preview.png")  

        try:
            print("Paused: Press Enter to continue...")
        except EOFError:
            pass  # Avoid crash if running without stdin (e.g. mpiexec with no terminal)


    # No piecewise mask
    def normalized_mask_optimized(d_perp,Re_tau):
        """
        H1 “optimal” erf mask with zero displacement length (O(ε^2) error):

          mask(d_perp) = ½ [1 − erf(√π · d_perp / δ*)] · mask_const

        where δ* = 3.11346786·ε, ε = √(1/Re).
        """
         # Compute damping length and optimal smoothing
        c = 2.5
        mask_const = Re_tau/c
        eta       = c*(1.0/Re_tau)
        #gamma      = np.sqrt((1.0 / Re) * (1.0 / mask_const_ref))
        gamma      = np.sqrt((eta/ Re_tau))
        delta_star = 3.11346786 * gamma
        
        #delta_star = 3.80171928 * gamma; # adjusted value by Burns

        return 0.5 * (1 - erf(np.sqrt(np.pi) * d_perp / delta_star)) * mask_const


    # Apply the function to compute the normalized mask
    final_mask_2d = normalized_mask_optimized(d_perp,Re_tau)


    print(final_mask_2d.shape)


    comm = MPI.COMM_WORLD
    rank = comm.rank

    gathered_mask_norm = comm.gather(final_mask_2d, root=0)
    gathered_x = comm.gather(np.transpose(x[:, :, 0]), root=0)  # shape (ny_local, nx)
    gathered_y = comm.gather(np.transpose(y[:, :, 0]), root=0)  # shape (ny_local, nx)

    if rank == 0:
        print("This is rank 0 â€” safe to do plotting or file I/O here.")
        print("gathered_mask_norm length:", len(gathered_mask_norm))
        print("gathered_x[0] shape:", gathered_x[0].shape)
        print("gathered_y[0] shape:", gathered_y[0].shape)
        

    if rank == 0:
        full_mask = np.concatenate(gathered_mask_norm, axis=0)  # shape: (ny_total, nx)
        full_y = np.concatenate(gathered_y, axis=0)              # shape: (ny_total, nx)
        full_x = gathered_x[0]                                   # all ranks have same x

        # Create meshgrid for plotting
        full_x, full_y = np.meshgrid(full_x.flatten(), full_y.flatten())

        print("full_mask shape:", full_mask.shape)
        print("full_x shape:", full_x.shape)
        print("full_y shape:", full_y.shape)
        print("full mask max:", np.max(full_mask))
        print("full mask min:", np.min(full_mask))

        plt.figure(figsize=(8, 6))
        plt.pcolormesh(full_x, full_y, full_mask, shading='auto', cmap='coolwarm')
        plt.colorbar(label="Mask Value")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Normalized Mask for Solid/Fluid Interface (MPI)")
        plt.tight_layout()
        plt.grid(False)
        plt.savefig("normalized_mask_preview.png", dpi=150)
        print("âœ… Saved plot to normalized_mask_preview.png")

        time.sleep(2)  # Optional: short pause
    # ---------- Step 4: Assign to Dedalus Field ----------
    # Expand to 3D and transpose to match Dedalus (nx, ny, nz)
    final_mask = np.repeat(final_mask_2d[:, :, np.newaxis], nz, axis=2)

    # Step 3: Transpose to match Dedalus (nx, ny, nz)
    final_mask = final_mask.transpose(1, 0, 2)

    mask['g'] = final_mask  # Assign computed mask to Dedalus field

    ##visualizing mask field
    assert mask['g'].shape == final_mask.shape, f"[Rank {rank}] Shape mismatch!"

    if np.isnan(mask['g']).any():
        raise ValueError(f"[Rank {rank}] NaNs detected in mask field!")

    if not np.isfinite(mask['g']).all():
        raise ValueError(f"[Rank {rank}] Inf or invalid values in mask field!")

    #local_min = np.min(mask['g'])
    #local_max = np.max(mask['g'])

    # Guard against empty local arrays
    if mask['g'].size > 0:
        local_min = np.min(mask['g'])
        local_max = np.max(mask['g'])
    else:
        local_min =  np.inf   # so it doesn’t affect the MPI MIN
        local_max = -np.inf   # so it doesn’t affect the MPI MAX

    global_min = comm.allreduce(local_min, op=MPI.MIN)
    global_max = comm.allreduce(local_max, op=MPI.MAX)

    if rank == 0:
        print(f"✅ Mask field successfully assigned across ranks.")
        print(f"Global mask value range: min={global_min}, max={global_max}")
        #input("stop here ..")


problem = d3.IVP([p, u, T, tau_p, tau_u1, tau_u2, tau_T1, tau_T2], namespace=locals())
problem.namespace.update({'t':problem.time})
problem.namespace.update({'T_s': 1.0})  # Base Temperature
problem.add_equation("trace(grad_u) + tau_p = 0")
#problem.add_equation("dt(u) - 1/Re_tau*div(grad_u) + grad(p) + lift(tau_u2) =-dPdx*(1+A*sin(omega*(t-load_time)))*ex -dot(u,grad(u))")
problem.add_equation("dt(u) - 1/Re_tau*div(grad_u) + grad(p) + lift(tau_u2) =-dPdx*(1+A*sin(omega*(t-load_time)))*ex -dot(u,grad(u)) - mask*u")
#problem.add_equation("dt(T) - 1/(Re_tau*Pr)*div(grad_T) + lift(tau_T2) = - u@grad(T) + (u@ex)/vol_average(u@ex)")
problem.add_equation("dt(T) - 1/(Re_tau*Pr)*div(grad_T) + lift(tau_T2) = - u@grad(T) + (u@ex)/vol_average(u@ex) - mask*(T - T_s)")


#problem.add_equation("u(y=-1) = 0") # change from -1 to -0.5
#problem.add_equation("u(y=+1) = 0") #change from 1 to 0.5
problem.add_equation("u(y=-Ly/2) = 0") # change from -1 to -0.5
problem.add_equation("u(y=+Ly/2) = 0") # change from 1 to 0.5
problem.add_equation("integ(p) = 0")
#problem.add_equation("T(y=+1)=A0+A*sin(omega*t)")
problem.add_equation("T(y=+Ly/2)=0")
problem.add_equation("T(y=-Ly/2)=0")

# Build Solver
dt = 0.0001 # 0.001
stop_sim_time = 200
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
    
if not restart:
    u['g'][0]=U_plus[np.newaxis, :, np.newaxis]+np.random.randn(*u['g'][0].shape) * 1e-6*np.sin(np.pi*(y+1)*0.5)
    T['g']=u['g'][0]
    file_handler_mode = 'overwrite'
else:
    write, initial_timestep = solver.load_state(checkpoint_path)
    file_handler_mode = 'append'


#This is random noise to trigger transition to turbulence
#+ np.random.randn(*u['g'][0].shape) * 1e-6*np.sin(np.pi*(y+1)*0.5) # Laminar solution (plane Poiseuille)+  random perturbation

#Full all 3D variables, every sim_dt=10, also serve as a checkpoint
snapshots = solver.evaluator.add_file_handler('snapshots_channel', sim_dt=3, max_writes=400, mode=file_handler_mode)
for field in solver.state:
    snapshots.add_task(field)

#2D slicing from the 3D data, every sim_dt=1
snapshots_2D = solver.evaluator.add_file_handler('snapshots_channel_2D',sim_dt=0.2,max_writes=4000, mode=file_handler_mode)
snapshots_2D.add_task(u(x=0), name='u_yz')
snapshots_2D.add_task(u(z=0), name='u_xy')
snapshots_2D.add_task(u(y=0), name='u_xz_mid')
snapshots_2D.add_task(u(y=(-1+5/Re_tau)), name='u_xz_viscous')
snapshots_2D.add_task(u(y=(-1+15/Re_tau)), name='u_xz_buffer')
snapshots_2D.add_task(u(y=(-1+50/Re_tau)), name='u_xz_log')

snapshots_2D.add_task(T(x=0), name='T_yz')
snapshots_2D.add_task(T(z=0), name='T_xy')
snapshots_2D.add_task(T(y=0), name='T_xz_mid')
snapshots_2D.add_task(T(y=(-1+5/Re_tau)), name='T_xz_viscous')
snapshots_2D.add_task(T(y=(-1+15/Re_tau)), name='T_xz_buffer')
snapshots_2D.add_task(T(y=(-1+50/Re_tau)), name='T_xz_log')

#1D statistics, every sim_dt=0.1
snapshots_stress = solver.evaluator.add_file_handler('snapshots_channel_stress', sim_dt=0.01, max_writes=40000,mode=file_handler_mode)

snapshots_stress.add_task(xz_average(u)@ex,name = 'u_bar')
snapshots_stress.add_task(d3.grad(xz_average(u)@ex)@ey,name = 'dudy')
snapshots_stress.add_task(vol_average(u@ex),name = 'u_bulk')
snapshots_stress.add_task(xz_average(((u-xz_average(u))@ex)**2),name = 'u_prime_u_prime')
snapshots_stress.add_task(xz_average(((u-xz_average(u))@ey)**2),name = 'v_prime_v_prime')
snapshots_stress.add_task(xz_average(((u-xz_average(u))@ez)**2),name = 'w_prime_w_prime')
snapshots_stress.add_task(xz_average(((u-xz_average(u))@ex)*(u-xz_average(u))@ey),name = 'u_prime_v_prime')

snapshots_stress.add_task(xz_average(T),name = 'T_bar')
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