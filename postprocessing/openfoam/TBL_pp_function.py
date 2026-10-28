import numpy as np
import scipy as sc
import pyvista as pv
import h5py
import os
import csv
from matplotlib import pyplot as plt
import pysemtools.interpolation.utils as interp_utils
#%% geometric stretching between 0 and xmax with a maximum dx



def geometric_stretching(dx0,stretching,dxmax,xmax):

    dx_base     = dx0 * (stretching**np.linspace(0,1000,1001))
    dx_base     = np.minimum(dx_base,dxmax)
    x_base      = np.cumsum( dx_base )
    N_inrange   = np.sum( x_base<=xmax )
    x_base      = x_base[0:N_inrange]
    x_base      = x_base/np.max(x_base) * xmax
    x                = np.zeros(N_inrange+1)
    x[0]             = 0.0
    x[1:N_inrange+1] = x_base

    return x

    #def create_curved_geometry():
        

    #%% define interpolation points
def user_defined_interpolating_points_plane():
    print("generate interpolation points")

    # Create the coordinates of the plane you want
    Lx_bbox = [0.0, 325]
    nx  = 192 * 7
    nz  = 1

    # add one point to exclude since periodic
    x = np.linspace( Lx_bbox[0] , Lx_bbox[1] , nx )
    z  = 0.0

    delta_nu    = 0.05
    dr0         = 1/180   # presuambly delta_nu=0.05
    stretching  = 1.02
    drmax       = 10*delta_nu  
    rmax        = 20

    y_base      = geometric_stretching(dr0,stretching,drmax,rmax)
    y_base[0]   = 0.001*delta_nu    # to avoid potential issues when maping to the exact walls
    y           = y_base
    print('x = ',x)
    print('y = ',y)
    print('z = ',z)
    delta_nu = 0.02
    yplus = (y[:])/delta_nu
    y_plus100 = yplus[:]
    delta_yplus = np.diff(y_plus100)
    print('yplus:',y_plus100)
    print('delta yplus:', delta_yplus)
    ny          = y.size
    Nstruct     = [ny, nx, nz]
    print('Nstruct=', Nstruct)

    Y,X,Z = np.meshgrid(y, x, z, indexing="ij")
    grid = pv.StructuredGrid(X,Y,Z)
    grid.save("interpolationPoints.vts")
    xyz = interp_utils.transform_from_array_to_list(ny, nx, nz, [X, Y, Z])

    return xyz, Nstruct

def user_defined_interpolating_points_lowCurv():
    print("generate interpolation points")

    # Geometry constants copied verbatim from the R100_deg30_rezoned_struct .geo so
    # the interpolation surface (flat plate -> circular arc (30 deg turn) -> straight
    # tail along the arc exit tangent) coincides with the mesh construction.
    # NOTE: the .geo uses a *truncated* PI = 3.142 (alpha = R*PI/6 / R = PI/6), which
    # differs from np.pi/6 by ~7e-5 rad and shifts the tail end by ~0.03; we copy the
    # same 3.142 so the arc/tail line up with the mesh instead of drifting off it.
    PI           = 3.142
    R            = 3050.0
    plane_length = 325.0
    delta_99     = 7.0
    height       = delta_99 * 7    # 24.5 : mesh wall-normal extent (top boundary)
    alpha        = PI / 10.0          # total arc turn angle (s_curve = R*PI/6 in .geo)
    arc_length   = R * alpha
    tail_length  = 325

    # Keep every probe strictly INSIDE the mesh outer boundaries. A point that lands
    # exactly on (or a hair past) a mesh face is reported "not found" by the point
    # search (rst just outside [-1, 1]) -- that was the entire top row (sat at y =
    # height), the outflow column (sat on the outflow face) and the curved wall row.
    # These small physical-unit margins pull those three edges just inside the mesh.
    top_margin = 0.5     # wall-normal inset below the top boundary (height)
    out_margin = 1.0     # streamwise inset before the outflow face
    wall_lift  = 0.005   # lift the whole profile off the wall (clears the curved
                         # wall face, where the point-in-element search is finicky)

    s_flat_end  = plane_length
    s_curve_end = s_flat_end + arc_length
    s_min = 0.0
    s_max = s_curve_end + tail_length - out_margin   # stop just short of the outflow

    pol_order = 6
    # keep the same point density (points per unit arclength) as
    # user_defined_interpolating_points_angle, just extended over the tail
    density = (192 * pol_order * 2) / (325.0 + 319.395)
    nx = int(round(density * s_max))
    nz = 1

    s = np.linspace(s_min, s_max, nx)
    z = 0.0

    ## Wall-normal distances (unchanged)
    fluid_nu = 0.0007
    u_tau = 0.0474954
    delta_nu = fluid_nu/ u_tau
    dr0 = 0.0007  # presumably delta_nu=0.05
    #stretching = 1.08
    stretching = 1.04
    #drmax = 11 * delta_nu
    drmax = 6.5 * delta_nu
    rmax = height - top_margin        # outermost station sits below the mesh top
    y_base = geometric_stretching(dr0, stretching, drmax, rmax)
    y_base = y_base + wall_lift       # lift the whole profile off the wall (was 0 at j=0)
    ny = y_base.size

    # Initialize arrays for structured grid
    X = np.zeros((ny, nx))
    Y = np.zeros((ny, nx))
    Z = np.zeros((ny, nx))

    k = alpha / arc_length  # curvature rate through the arc section (= 1/R)

    # tangent/normal at the end of the arc, reused as the constant tail direction
    x_arc_end = s_flat_end + (1 / k) * np.sin(alpha)
    y_arc_end = (1 / k) * (np.cos(alpha) - 1)

    # Generate points
    for i in range(nx):
        s_i = s[i]
        if s_i <= s_flat_end:
            # Straight part: position (s_i, 0), normal (0, 1)
            x_surf = s_i
            y_surf = 0.0
            n_x = 0.0
            n_y = 1.0
        elif s_i <= s_curve_end:
            # Curved part: smooth transition with linearly increasing angle
            ds = s_i - s_flat_end
            theta_i = k * ds
            # Integrated position for Euler spiral (clothoid)
            x_surf = s_flat_end + (1 / k) * np.sin(theta_i)
            y_surf = (1 / k) * (np.cos(theta_i) - 1)
            # Normal vector (perpendicular to tangent)
            n_x = np.sin(theta_i)
            n_y = np.cos(theta_i)
        else:
            # Straight tail: fixed angle alpha, extending along the tangent
            # direction at the end of the arc (matches dx_tail/dy_tail in the .geo)
            ds_tail = s_i - s_curve_end
            x_surf = x_arc_end + ds_tail * np.cos(alpha)
            y_surf = y_arc_end - ds_tail * np.sin(alpha)
            n_x = np.sin(alpha)
            n_y = np.cos(alpha)

        # Offset points in wall-normal direction
        for j in range(ny):
            X[j, i] = x_surf + y_base[j] * n_x
            Y[j, i] = y_surf + y_base[j] * n_y
            Z[j, i] = z

    # Debugging prints (adjusted for new coordinates)
    print('s = ', s)
    print('y_base = ', y_base)
    print('z = ', z)
    #delta_nu = 0.02
    yplus = y_base / delta_nu
    delta_yplus = np.diff(yplus)
    print('yplus:', yplus)
    print('delta yplus:', delta_yplus)
    Nstruct = [ny, nx, nz]
    print('Nstruct=', Nstruct)
    grid = pv.StructuredGrid(X,Y,Z)
    grid.save("interpolationPoints.vts")
    # Transform to list format
    print(X.shape, Y.shape, Z.shape)
    xyz = interp_utils.transform_from_array_to_list(ny, nx, nz, [X, Y, Z])

    return xyz, Nstruct

def user_defined_interpolating_points_highCurv():
    print("generate interpolation points")

    # Geometry constants copied verbatim from the R100_deg30_rezoned_struct .geo so
    # the interpolation surface (flat plate -> circular arc (30 deg turn) -> straight
    # tail along the arc exit tangent) coincides with the mesh construction.
    # NOTE: the .geo uses a *truncated* PI = 3.142 (alpha = R*PI/6 / R = PI/6), which
    # differs from np.pi/6 by ~7e-5 rad and shifts the tail end by ~0.03; we copy the
    # same 3.142 so the arc/tail line up with the mesh instead of drifting off it.
    PI           = 3.142
    R            = 610.0
    plane_length = 325.0
    delta_99     = 7.0
    height       = delta_99 * 5    # 24.5 : mesh wall-normal extent (top boundary)
    alpha        = PI / 2.0          # total arc turn angle (s_curve = R*PI/6 in .geo)
    arc_length   = R * alpha
    tail_length  = 162.5

    # Keep every probe strictly INSIDE the mesh outer boundaries. A point that lands
    # exactly on (or a hair past) a mesh face is reported "not found" by the point
    # search (rst just outside [-1, 1]) -- that was the entire top row (sat at y =
    # height), the outflow column (sat on the outflow face) and the curved wall row.
    # These small physical-unit margins pull those three edges just inside the mesh.
    top_margin = 0.5     # wall-normal inset below the top boundary (height)
    out_margin = 1.0     # streamwise inset before the outflow face
    wall_lift  = 0.005   # lift the whole profile off the wall (clears the curved
                         # wall face, where the point-in-element search is finicky)

    s_flat_end  = plane_length
    s_curve_end = s_flat_end + arc_length
    s_min = 0.0
    s_max = s_curve_end + tail_length - out_margin   # stop just short of the outflow

    pol_order = 7
    # keep the same point density (points per unit arclength) as
    # user_defined_interpolating_points_angle, just extended over the tail
    density = (192 * pol_order * 2) / (325.0 + 319.395)
    nx = int(round(density * s_max))
    nz = 1

    s = np.linspace(s_min, s_max, nx)
    z = 0.0

    ## Wall-normal distances (unchanged)
    fluid_nu = 0.0007
    u_tau = 0.0474954
    delta_nu = fluid_nu/ u_tau
    dr0 = 0.0007  # presumably delta_nu=0.05
    #stretching = 1.08
    stretching = 1.10
    #drmax = 11 * delta_nu
    drmax = 6.5 * delta_nu
    rmax = height - top_margin        # outermost station sits below the mesh top
    y_base = geometric_stretching(dr0, stretching, drmax, rmax)
    y_base = y_base + wall_lift       # lift the whole profile off the wall (was 0 at j=0)
    ny = y_base.size

    # Initialize arrays for structured grid
    X = np.zeros((ny, nx))
    Y = np.zeros((ny, nx))
    Z = np.zeros((ny, nx))

    k = alpha / arc_length  # curvature rate through the arc section (= 1/R)

    # tangent/normal at the end of the arc, reused as the constant tail direction
    x_arc_end = s_flat_end + (1 / k) * np.sin(alpha)
    y_arc_end = (1 / k) * (np.cos(alpha) - 1)

    # Generate points
    for i in range(nx):
        s_i = s[i]
        if s_i <= s_flat_end:
            # Straight part: position (s_i, 0), normal (0, 1)
            x_surf = s_i
            y_surf = 0.0
            n_x = 0.0
            n_y = 1.0
        elif s_i <= s_curve_end:
            # Curved part: smooth transition with linearly increasing angle
            ds = s_i - s_flat_end
            theta_i = k * ds
            # Integrated position for Euler spiral (clothoid)
            x_surf = s_flat_end + (1 / k) * np.sin(theta_i)
            y_surf = (1 / k) * (np.cos(theta_i) - 1)
            # Normal vector (perpendicular to tangent)
            n_x = np.sin(theta_i)
            n_y = np.cos(theta_i)
        else:
            # Straight tail: fixed angle alpha, extending along the tangent
            # direction at the end of the arc (matches dx_tail/dy_tail in the .geo)
            ds_tail = s_i - s_curve_end
            x_surf = x_arc_end + ds_tail * np.cos(alpha)
            y_surf = y_arc_end - ds_tail * np.sin(alpha)
            n_x = np.sin(alpha)
            n_y = np.cos(alpha)

        # Offset points in wall-normal direction
        for j in range(ny):
            X[j, i] = x_surf + y_base[j] * n_x
            Y[j, i] = y_surf + y_base[j] * n_y
            Z[j, i] = z

    # Debugging prints (adjusted for new coordinates)
    print('s = ', s)
    print('y_base = ', y_base)
    print('z = ', z)
    #delta_nu = 0.02
    yplus = y_base / delta_nu
    delta_yplus = np.diff(yplus)
    print('yplus:', yplus)
    print('delta yplus:', delta_yplus)
    Nstruct = [ny, nx, nz]
    print('Nstruct=', Nstruct)
    grid = pv.StructuredGrid(X,Y,Z)
    grid.save("interpolationPoints.vts")
    # Transform to list format
    print(X.shape, Y.shape, Z.shape)
    xyz = interp_utils.transform_from_array_to_list(ny, nx, nz, [X, Y, Z])

    return xyz, Nstruct

def user_defined_interpolating_points_withTail():
    print("generate interpolation points")

    # Geometry constants copied verbatim from the R100_deg30_rezoned_struct .geo so
    # the interpolation surface (flat plate -> circular arc (30 deg turn) -> straight
    # tail along the arc exit tangent) coincides with the mesh construction.
    # NOTE: the .geo uses a *truncated* PI = 3.142 (alpha = R*PI/6 / R = PI/6), which
    # differs from np.pi/6 by ~7e-5 rad and shifts the tail end by ~0.03; we copy the
    # same 3.142 so the arc/tail line up with the mesh instead of drifting off it.
    PI           = 3.142
    R            = 610.0
    plane_length = 325.0
    delta_99     = 7.0
    height       = delta_99 * 3.5    # 24.5 : mesh wall-normal extent (top boundary)
    alpha        = PI / 6.0          # total arc turn angle (s_curve = R*PI/6 in .geo)
    arc_length   = R * alpha
    tail_length  = 162.5

    # Keep every probe strictly INSIDE the mesh outer boundaries. A point that lands
    # exactly on (or a hair past) a mesh face is reported "not found" by the point
    # search (rst just outside [-1, 1]) -- that was the entire top row (sat at y =
    # height), the outflow column (sat on the outflow face) and the curved wall row.
    # These small physical-unit margins pull those three edges just inside the mesh.
    top_margin = 0.5     # wall-normal inset below the top boundary (height)
    out_margin = 1.0     # streamwise inset before the outflow face
    wall_lift  = 0.005   # lift the whole profile off the wall (clears the curved
                         # wall face, where the point-in-element search is finicky)

    s_flat_end  = plane_length
    s_curve_end = s_flat_end + arc_length
    s_min = 0.0
    s_max = s_curve_end + tail_length - out_margin   # stop just short of the outflow

    pol_order = 7
    # keep the same point density (points per unit arclength) as
    # user_defined_interpolating_points_angle, just extended over the tail
    density = (192 * pol_order * 2) / (325.0 + 319.395)
    nx = int(round(density * s_max))
    nz = 1

    s = np.linspace(s_min, s_max, nx)
    z = 0.0

    ## Wall-normal distances (unchanged)
    fluid_nu = 0.0007
    u_tau = 0.0474954
    delta_nu = fluid_nu/ u_tau
    dr0 = 0.0007  # presumably delta_nu=0.05
    #stretching = 1.08
    stretching = 1.04
    #drmax = 11 * delta_nu
    drmax = 6.5 * delta_nu
    rmax = height - top_margin        # outermost station sits below the mesh top
    y_base = geometric_stretching(dr0, stretching, drmax, rmax)
    y_base = y_base + wall_lift       # lift the whole profile off the wall (was 0 at j=0)
    ny = y_base.size

    # Initialize arrays for structured grid
    X = np.zeros((ny, nx))
    Y = np.zeros((ny, nx))
    Z = np.zeros((ny, nx))

    k = alpha / arc_length  # curvature rate through the arc section (= 1/R)

    # tangent/normal at the end of the arc, reused as the constant tail direction
    x_arc_end = s_flat_end + (1 / k) * np.sin(alpha)
    y_arc_end = (1 / k) * (np.cos(alpha) - 1)

    # Generate points
    for i in range(nx):
        s_i = s[i]
        if s_i <= s_flat_end:
            # Straight part: position (s_i, 0), normal (0, 1)
            x_surf = s_i
            y_surf = 0.0
            n_x = 0.0
            n_y = 1.0
        elif s_i <= s_curve_end:
            # Curved part: smooth transition with linearly increasing angle
            ds = s_i - s_flat_end
            theta_i = k * ds
            # Integrated position for Euler spiral (clothoid)
            x_surf = s_flat_end + (1 / k) * np.sin(theta_i)
            y_surf = (1 / k) * (np.cos(theta_i) - 1)
            # Normal vector (perpendicular to tangent)
            n_x = np.sin(theta_i)
            n_y = np.cos(theta_i)
        else:
            # Straight tail: fixed angle alpha, extending along the tangent
            # direction at the end of the arc (matches dx_tail/dy_tail in the .geo)
            ds_tail = s_i - s_curve_end
            x_surf = x_arc_end + ds_tail * np.cos(alpha)
            y_surf = y_arc_end - ds_tail * np.sin(alpha)
            n_x = np.sin(alpha)
            n_y = np.cos(alpha)

        # Offset points in wall-normal direction
        for j in range(ny):
            X[j, i] = x_surf + y_base[j] * n_x
            Y[j, i] = y_surf + y_base[j] * n_y
            Z[j, i] = z

    # Debugging prints (adjusted for new coordinates)
    print('s = ', s)
    print('y_base = ', y_base)
    print('z = ', z)
    #delta_nu = 0.02
    yplus = y_base / delta_nu
    delta_yplus = np.diff(yplus)
    print('yplus:', yplus)
    print('delta yplus:', delta_yplus)
    Nstruct = [ny, nx, nz]
    print('Nstruct=', Nstruct)
    grid = pv.StructuredGrid(X,Y,Z)
    grid.save("interpolationPoints.vts")
    # Transform to list format
    print(X.shape, Y.shape, Z.shape)
    xyz = interp_utils.transform_from_array_to_list(ny, nx, nz, [X, Y, Z])

    return xyz, Nstruct


def calculate_TBL_quantities( x , y , UVW ):
    import numpy as np
    import sys
    
    if np.ndim(UVW)>3:
        sys.exit('UVW is supposed to a 3D array')
    if UVW.shape[0] != np.size(y) or UVW.shape[1] != np.size(x) or UVW.shape[2]!=3:
        sys.exit('UVW is supposed to be Ny*Nx*3')
   # if np.abs(np.max( UVW[:,:,0] )-1)>=.01:
    #    sys.exit('uvw is supposed to be normalized to have an ideal value of 1')

    dStar           = np.zeros_like(x)
    theta           = np.zeros_like(x)
    delta99         = np.zeros_like(x)
    delta99_approx  = np.zeros_like(x)
    dStar_trunc     = np.zeros_like(x)
    theta_trunc     = np.zeros_like(x)
    Uinf            = np.zeros_like(x)
    
    ny = np.size(y)

    def _first_crossing(u_col, thr):
        """First wall-normal location (index) where u_col reaches thr, scanning
        outward from the wall. Returns None if the threshold is never reached
        (e.g. boundary layer thicker than the sampled wall-normal height)."""
        above = np.nonzero(u_col >= thr)[0]
        if above.size == 0:
            return None
        return above[0]

    n_incomplete = 0
    for ix in range( np.size(x) ):
        layer99 = .99
        # Bootstrap the local edge velocity from the profile itself instead of assuming
        # the free stream sits at the absolute level 1. Not every case is normalized to
        # U_ref = 1 (R500_deg18 runs at U_e ~ 0.90-0.94), and even in a normalized case
        # U_e varies by a few percent along a curved wall -- an absolute 0.99 threshold
        # then never triggers and every station silently falls into the NaN branch below.
        # This is only a first guess for delta99; Uinf is refined further down.
        u_ref_guess         = np.max( UVW[:,ix,0] )
        # first crossing of 0.99*U_e, scanning outward from the wall
        i99                 = _first_crossing( UVW[:,ix,0] , layer99*u_ref_guess )
        if i99 is None:
            # U never recovers to 0.99*U_ref within the sampled wall-normal height at
            # this station (grid too short for the local BL, or separated flow) --
            # nothing downstream of here is computable, so mark it explicitly rather
            # than silently deriving Uinf from a zero-width slice at the domain edge.
            delta99_approx[ix] = np.nan
            Uinf[ix]           = np.nan
            delta99[ix]        = np.nan
            dStar[ix]          = np.nan
            theta[ix]          = np.nan
            dStar_trunc[ix]    = np.nan
            theta_trunc[ix]    = np.nan
            n_incomplete += 1
            continue

        delta99_approx[ix]  = y[i99]
        #print('i99, y_i99', i99, y[i99])
        y_2delta99          = 2*delta99_approx[ix] # hier faktor 2 weglassen
        i_inf               = np.argmin( np.abs( y - y_2delta99 ) )
        i_inf               = min(i_inf, ny - 2)  # keep >=2 points so the trapz below isn't a zero-width slice
        uinf_here       = np.trapezoid( UVW[i_inf:,ix,0] , y[i_inf:] ) / np.trapezoid( np.ones_like(y[i_inf:]) , y[i_inf:] )
        Uinf[ix]        = uinf_here
        # delta99: linear interpolation across the first crossing of 0.99*Uinf
        j               = _first_crossing( UVW[:,ix,0] , layer99*uinf_here )
        if j is None:
            j = ny - 1
        if j == 0:
            delta99[ix] = y[0]
        else:
            u0, u1 = UVW[j-1,ix,0], UVW[j,ix,0]
            delta99[ix] = y[j-1] + (layer99*uinf_here - u0) * (y[j]-y[j-1]) / (u1-u0)
       # print('delta99', delta99[ix])
        uUinf           = UVW[:,ix,0]/uinf_here
        dStar[ix]       = np.trapezoid( 1-uUinf , y, axis=0 )
        theta[ix]       = np.trapezoid( uUinf*(1-uUinf) , y, axis=0 )

        y_trunc         = y[:i_inf+1]
        u_trunc         = UVW[:i_inf+1,ix,0]
        uinf_trunc      = u_trunc[-1]
        dStar_trunc[ix] = np.trapezoid( 1-u_trunc/uinf_trunc , y_trunc)
        theta_trunc[ix] = np.trapezoid( u_trunc/uinf_trunc*(1-u_trunc/uinf_trunc) , y_trunc)

    if n_incomplete > 0:
        print(f"calculate_TBL_quantities: {n_incomplete}/{np.size(x)} streamwise "
              f"stations never reached 0.99*max(U) within the sampled wall-normal "
              f"height -- delta99/dStar/theta/Re_* are NaN there (column is all-NaN, "
              f"e.g. interpolation points that fell outside the mesh).")

    H12     = dStar / theta


    return delta99,delta99_approx ,  dStar,dStar_trunc , theta,theta_trunc , Uinf, H12
                
                
def wall_gradient_from_sublayer(y, Us, fluid_nu, yplus_max=2.0, min_points=4):
    """dU_s/dn at the wall, taken from the resolved viscous sublayer of the velocity
    profile instead of from the interpolated gradient field.

    Fits  U_s(n) = a*n + c*n^2  with no-slip enforced (no constant term) over the
    points inside y+ < yplus_max, and returns the slope a. The quadratic term absorbs
    the slight curvature of the sublayer so the answer does not drift with how far
    out the window happens to reach.

    The window needs y+, which needs u_tau, which is what we are solving for, so it
    is iterated -- it converges in 2-3 passes. If fewer than min_points fall inside
    the window (coarse near-wall grid), the first min_points are used instead.

    yplus_max=2 was picked by benchmarking against the stored gradient on FLAT-plate
    stations, where that gradient is exact (undeformed elements). Calibration:

        form / window        flat: fitted/exact        bend: U+/y+ at first point
        a*n + c*n^2, y+<4      1.0120 +- 0.0032           0.9892 +- 0.0017
        a*n + c*n^2, y+<2      1.0035 +- 0.0010           0.9978 +- 0.0006   <- chosen
        a*n,         y+<2      0.9935 +- 0.0024           1.0042 +- 0.0031
        a*n,         y+<1      0.9965 +- 0.0010           1.0018 +- 0.0022
        a*n + d*n^4, y+<5      0.9921 +- 0.0040           1.0056 +- 0.0040

    (R100_deg30 restart8; both must be 1.0. The a*n/y+<1 variant scores similarly but
    collapses onto the min_points floor on that grid -- only 3 points -- so the window
    stops controlling it. The chosen form uses 4-5 points there and 20-29 on R500_deg18.)
    """
    a = Us[1] / y[1] if y[1] > 0 else Us[0] / y[0]      # first guess
    for _ in range(20):
        u_tau_it = np.sqrt(np.abs(fluid_nu * a))
        if not np.isfinite(u_tau_it) or u_tau_it <= 0:
            break
        m = y <= yplus_max * fluid_nu / u_tau_it
        if m.sum() < min_points:
            m = np.zeros_like(y, dtype=bool)
            m[:min_points] = True
        A = np.stack([y[m], y[m]**2], axis=1)
        a_new = np.linalg.lstsq(A, Us[m], rcond=None)[0][0]
        if not np.isfinite(a_new):
            break
        converged = abs(a_new - a) <= 1e-10 * abs(a)
        a = a_new
        if converged:
            break
    return a


def calculate_friction_units(Rer, dUidXj, y=None, UVW_rot=None, method='fit'):
    """method='fit'      -> wall gradient from the resolved sublayer (needs y, UVW_rot)
       method='gradient' -> ORIGINAL behaviour: stored dUidXj at the first point

    WHY 'fit' is the default (set method='gradient' to get the old numbers back):
    dUidXj is Neko's *pre-differentiated* field interpolated onto this point cloud, and
    it is only C0-continuous across element faces. The wall IS an element face, so in
    curved-wall regions the interpolated gradient is badly wrong in the first 2-3 points
    off the wall -- measured deficits of 15-37% for R100_deg30, while the same points'
    VELOCITY is perfectly clean (it extrapolates to U=0 at n=0 and is linear). From
    y+ ~ 3 outward the stored gradient agrees with a finite difference to <1%; it is
    only the wall-most points, exactly where u_tau is read, that are corrupted.
    Effect of the old path in the bend: u_tau ~25% low, c_f ~50-60% low, and the U+
    profiles sit up to 50% ABOVE the U+ = y+ line in the viscous sublayer (because
    u_tau low by a factor a puts y+ low by a and U+ high by 1/a, i.e. U+/y+ = 1/a^2).
    Flat-plate stations are unaffected -- both routes agree there to 4 digits.
    """
    fluid_nu        = 1/Rer

    if method == 'fit' and y is not None and UVW_rot is not None:
        y  = np.asarray(y, dtype=np.float64)
        nx = UVW_rot.shape[1]
        dUsdn = np.array([wall_gradient_from_sublayer(y, UVW_rot[:, ix, 0, 0].astype(np.float64),
                                                      fluid_nu)
                          for ix in range(nx)])
    else:
        # ORIGINAL path. NOTE the storage convention of the (rotated) gradient tensor is
        # the TRANSPOSE of what the name suggests: dUidXj[..., a, b] = dU_b/dx_a, so slot
        # [1,0] is dU_s/dn. The repeated index is therefore deliberate, not a typo: do NOT
        # "repair" it into the strain rate 0.5*(dUidXj[...,0,1]+dUidXj[...,1,0]). At a
        # no-slip wall dU_n/ds = 0, so that symmetrised form is just half of dU_s/dn and
        # would under-predict u_tau by ~30%.
        dUsdn = (dUidXj[0,:,0,1,0] + dUidXj[0,:,0,1,0])/2

    tauW_over_rho   = fluid_nu * dUsdn
    u_tau           = np.sqrt(np.abs(tauW_over_rho))
    delta_nu        = fluid_nu / u_tau

    return fluid_nu, tauW_over_rho, u_tau, delta_nu

def calculate_reynolds_numbers(delta99, delta_nu, Uinf, fluid_nu, dStar, theta):
    Re_tau          = delta99        / delta_nu
    Re_delta99      = delta99 * Uinf / fluid_nu
    Re_dStar        = dStar   * Uinf / fluid_nu
    Re_theta        = theta   * Uinf / fluid_nu
    print('Re_tau = ', Re_tau )
    print('Re_delta99 = ', Re_delta99 )
    print('Re_d* = ', Re_dStar )
    print('Re_theta = ', Re_theta )
    
    return  Re_tau , Re_delta99, Re_dStar, Re_theta
    
def define_vectors(XYZ):
    x = XYZ[0,:,0,0]  
    nx = np.zeros(x.shape[0])
    v1 = np.zeros_like(XYZ)
    v2 = np.zeros_like(v1)
    v3 =np.zeros_like(v1) 
    v3[...,2] = 1.0

    x_dist = np.zeros_like(XYZ)
    y_dist = np.zeros_like(XYZ)
    for i in range(len(x)):
        if i == 0:
            v1[:,i,...] = XYZ[:,i+1,:,:] - XYZ[:,i,:,:]
     
        elif i == len(x)-1:
            v1[:,i,...] = XYZ[:,i,:,:] - XYZ[:,i-1,:,:]
        else:
            f = 0
            v1[:,i,...]  = 0.5 * (XYZ[:,i+1,...] - XYZ[:,i-1,...])
    
    v1 = v1/np.sqrt(np.sum(v1**2,axis=-1,keepdims=True))
    v2 = np.cross(v3,v1,axis=3) 
    v2 = v2/np.sqrt(np.sum(v2**2,axis=-1,keepdims=True))
    
    return v1, v2, v3
    
def define_cylindrical_coord_system_vectors(XYZ):

    # unit vector in direction 1 is along z
    v1 = np.zeros_like(XYZ) 
    v1[...,2] = 1.0

    # unit vector in direction 2 is along the wall normal
    # this is based on dimension one of XYZ corresponding to r
    v2 = XYZ[-1,...]-XYZ[0,...]
    v2 = v2/np.sqrt(np.sum(v2**2,axis=-1,keepdims=True))
    v2 = v2 * np.ones_like(v1)

    # the last vector in a right-handed orthogonal coordinate system is the cross product of the first two
    v3 = np.cross(v1,v2,axis=3)

    return v1,v2,v3

def find_index(desired_Retheta, Re_theta):
    indices = np.where(Re_theta >= desired_Retheta)[0]
    if len(indices) == 0:
        raise ValueError(f"Re_theta never reaches {desired_Retheta} in the array (max={Re_theta.max():.1f})")
    index = indices[0]
    print('Re theta 1000', Re_theta[index], index)
    return index

def extract_tke_at_xposition(index, u_tau, Rij_rot, y_indeces, U_ref=1.0):

    # Rij_rot is non-dimensional (normalized by the SEM reference velocity, where the
    # free-stream value is 1). Rescale to physical units (m^2/s^2) for OpenFOAM using
    # U_ref, the physical free-stream velocity chosen for the OpenFOAM case (must be
    # picked together with nu/L_ref so that U_ref*L_ref/nu reproduces Rer).
    tke = 0.5 * (Rij_rot[:, index, 0, 0]
               + Rij_rot[:, index, 0, 1]
               + Rij_rot[:, index, 0, 2]) * U_ref**2
    tke_resorted = tke[y_indeces]
    np.savetxt("tke_inlet.dat",
        np.column_stack([ tke_resorted]),
        header="tke", fmt="%.6f")
        
def extract_vel_at_xposition(index, UVW, y_indeces):

    uvw_at_Retheta = UVW[:, index,0]
    print('shape', uvw_at_Retheta.shape)
    u = uvw_at_Retheta[:,0]
    u_resorted = u[y_indeces]
    v = np.zeros_like(u)
    v_resorted = v[y_indeces]
    w = np.zeros_like(u)
    w_resorted = w[y_indeces]
    np.savetxt("velocity_inlet.dat",
            np.column_stack([ u_resorted, v_resorted, w_resorted]),
            header="u", fmt="(%.6f %.6f %.6f)")

   
def extract_omega_at_xposition( index, Rij_rot, Diss_ij_rot, u_tau, delta_nu,y_indeces):
      
    C_mu = 0.09              
    print('diss shape', Diss_ij_rot.shape)
    #tke = 1/2*(Rij_rot[:, index, 0,0]**2 +Rij_rot[:, index, 0,1]**2 +Rij_rot[:, index, 0,2]**2)/ u_tau[index]**2
    tke = 0.5 * (Rij_rot[:, index, 0, 0]
               + Rij_rot[:, index, 0, 1]
               + Rij_rot[:, index, 0, 2])
   # diss = (Diss_ij_rot[:, index, 0, 0] + Diss_ij_rot[:, index, 0, 1] + Diss_ij_rot[:, index, 0,2]+ 2*Diss_ij_rot[:, index, 0, 3]+ 2*Diss_ij_rot[:, index, 0, 4] + 2*Diss_ij_rot[:, index, 0, 5])/(u_tau[index]**3/delta_nu[index])
    #diss_div = 1/2 *(Diss_ij_rot[:, index, 0, 0]/(u_tau[index]**3/delta_nu[index]) + Diss_ij_rot[:, index, 0, 1]/(u_tau[index]**3/delta_nu[index]) + Diss_ij_rot[:, index, 0,2]/(u_tau[index]**3/delta_nu[index]))
    diss = (Diss_ij_rot[:, index, 0, 0]
          + Diss_ij_rot[:, index, 0, 1]
          + Diss_ij_rot[:, index, 0, 2])
    print(diss)
    omega =np.abs( diss / (C_mu*tke))
    omega_resorted = omega[y_indeces]
    print('omega', omega.shape)
    print('omega inlet',omega)
    np.savetxt("omega_inlet.dat",
            np.column_stack([ omega_resorted]),
            header="omega", fmt="%.6f")

    print(f"omega range: {omega.min():.4f} – {omega.max():.4f}")


def write_reference_style_profiles(out_dir, ix_list, y,
                                   delta99, delta_nu, u_tau, Uinf,
                                   Re_theta, Re_tau, Re_dStar, H12, s,
                                   UVW_rot, Rij_rot,
                                   Prod_ij_rot, Conv_ij_rot, Diss_ij_rot,
                                   PRS_ij_rot, PressTrans_ij_rot, TurbTrans_ij_rot,
                                   ViscDiff_ij_rot, Res_ij):

    os.makedirs(out_dir, exist_ok=True)

    comp_names = ['uu', 'vv', 'ww', 'uv']         # budget/stress component index 0..3

    for ix in ix_list:
        ReT     = Re_theta[ix]
        scale_b = u_tau[ix]**3 / delta_nu[ix]     # budget wall-unit scale
        yp      = y / delta_nu[ix]                 # y+
        yd99    = y / delta99[ix]                  # y/delta_99
        c_f     = 2.0 * u_tau[ix]**2 / Uinf[ix]**2

        header_int = (
            "% Curved-TBL data, rotated into the wall-aligned frame\n"
            f"% Re_theta   = {ReT:14.6f}\n"
            f"% Re_tau     = {Re_tau[ix]:14.6f}\n"
            f"% Re_dStar   = {Re_dStar[ix]:14.6f}\n"
            f"% H_12       = {H12[ix]:14.6f}\n"
            f"% c_f        = {c_f:14.8f}\n"
            f"% u_tau      = {u_tau[ix]:14.8f}\n"
            f"% delta_99   = {delta99[ix]:14.8f}\n"
            f"% s          = {s[ix]:14.6f}\n"
            "%"
        )

        # ---- velocity / rms profile -------------------------------------------
        Up     = UVW_rot[:, ix, 0, 0] / u_tau[ix]
        Vp     = UVW_rot[:, ix, 0, 1] / u_tau[ix]
        # sqrt(abs(...)) guards against small negative stresses from statistical noise
        urms   = np.sqrt(np.abs(Rij_rot[:, ix, 0, 0])) / u_tau[ix]
        vrms   = np.sqrt(np.abs(Rij_rot[:, ix, 0, 1])) / u_tau[ix]
        wrms   = np.sqrt(np.abs(Rij_rot[:, ix, 0, 2])) / u_tau[ix]
        uvp    = Rij_rot[:, ix, 0, 3] / u_tau[ix]**2
        dUpdyp = np.gradient(Up, yp)
        vel = np.column_stack([yd99, yp, Up, Vp, urms, vrms, wrms, uvp, dUpdyp])
        vel_header = header_int + \
            "\n% y/d99            y+               U+               V+" \
            "               urms+            vrms+            wrms+" \
            "            uv+              dU+/dy+"
        np.savetxt(os.path.join(out_dir, f"TBL_Re{ReT:.0f}_vel.prof.txt"),
                   vel, fmt="%18.9e", header=vel_header, comments='')

        # ---- budgets: one file per stress component (uu, vv, ww, uv) ----------
        velp = PRS_ij_rot + PressTrans_ij_rot     # velocity-pressure-gradient (combined)
        bud_header = header_int + \
            "\n% y/d99            y+               conv+            prod+" \
            "            diss+            t-diff+          velp+" \
            "            vis-diff+        residual+"
        for k, cname in enumerate(comp_names):
            cols = np.column_stack([
                yd99, yp,
                Conv_ij_rot[:, ix, 0, k]      / scale_b,
                Prod_ij_rot[:, ix, 0, k]      / scale_b,
                Diss_ij_rot[:, ix, 0, k]      / scale_b,
                TurbTrans_ij_rot[:, ix, 0, k] / scale_b,
                velp[:, ix, 0, k]             / scale_b,
                ViscDiff_ij_rot[:, ix, 0, k]  / scale_b,
                Res_ij[:, ix, 0, k]           / scale_b,
            ])
            np.savetxt(os.path.join(out_dir, f"TBL_Re{ReT:.0f}_bud_{cname}.txt"),
                       cols, fmt="%18.9e", header=bud_header, comments='')

        # ---- TKE (k) budget: 0.5 * trace over the three normal components -----
        def tke(t):
            return 0.5 * np.sum(t[:, ix, 0, 0:3], axis=-1)
        colsk = np.column_stack([
            yd99, yp,
            tke(Conv_ij_rot)      / scale_b,
            tke(Prod_ij_rot)      / scale_b,
            tke(Diss_ij_rot)      / scale_b,
            tke(TurbTrans_ij_rot) / scale_b,
            tke(velp)             / scale_b,
            tke(ViscDiff_ij_rot)  / scale_b,
            tke(Res_ij)           / scale_b,
        ])
        np.savetxt(os.path.join(out_dir, f"TBL_Re{ReT:.0f}_bud_k.txt"),
                   colsk, fmt="%18.9e", header=bud_header, comments='')

        print(f"wrote reference-style profiles for Re_theta={ReT:.0f} -> {out_dir}")


def write_beta_dpds(out_dir, s, dStar, tauW_over_rho, P_w, Re_theta, *,
                     Uinf, Re_tau, Re_dStar, Re_delta99, delta99, theta, H12,
                     u_tau, delta_nu, fname='beta_dpds.txt'):

    os.makedirs(out_dir, exist_ok=True)

    dPw_ds = np.gradient(P_w, s)
    beta = dStar / tauW_over_rho * dPw_ds

    col_names = ['s', 'Re_theta', 'Re_tau', 'Re_dStar', 'Re_delta99', 'dStar',
                 'delta99', 'theta', 'H12', 'tau_w_over_rho', 'u_tau',
                 'delta_nu', 'P_w', 'dPw_ds', 'beta', 'U_inf']
    cols = np.column_stack([s, Re_theta, Re_tau, Re_dStar, Re_delta99, dStar,
                             delta99, theta, H12, tauW_over_rho, u_tau,
                             delta_nu, P_w, dPw_ds, beta, Uinf])
    header = (
        "% Curved-TBL streamwise boundary-layer quantities (raw, unsmoothed)\n"
        "%\n"
        "% " + "".join(f"{name:<17s}" for name in col_names)
    )
    out_path = os.path.join(out_dir, fname)
    np.savetxt(out_path, cols, fmt="%18.9e", header=header, comments='')
    print(f"wrote beta/dp_ds profile -> {out_path}")

