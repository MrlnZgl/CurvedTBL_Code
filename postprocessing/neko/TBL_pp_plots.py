import numpy as np
import scipy as sc
import pyvista as pv
import h5py
import os
import re
import glob
from matplotlib import pyplot as plt
import plot_style  # applies all matplotlib rcParams / backend setup on import

# Master on/off switches: flip either to False to drop that literature source
# from every plot (budgets, fluctuations, RMS, near-wall u) without touching
# anything else below.
add_schlatter_reference = True
add_jimenez_reference = True

top_dir = '/Users/marleneziegler/Desktop/5. Semester/Masterarbeit/TBL/reference_data'

stats_dir        = top_dir

# --- reference-data loading -------------------------------------------------
# Each literature source can ship several Re_theta stations, one per folder
# (e.g. schlatter_ref_670, schlatter_ref_1000, schlatter_ref_1410). Datasets
# are discovered by globbing, and each one's actual Re_theta is parsed out of
# its own file header, so adding/removing a folder needs no code changes.

_RE_THETA_RE = re.compile(r'Re_\{\\?theta\}\s*=\s*([\d.]+)')


def _read_profile_table(path):
    # finds the column-header row (contains 'y+'), then skips any further
    # '%'-commented separator rows (Jimenez files have one, Schlatter doesn't)
    with open(path) as f:
        lines = f.readlines()
    start = next(i for i, l in enumerate(lines) if 'y+' in l) + 1
    while start < len(lines) and lines[start].lstrip().startswith('%'):
        start += 1
    rows = [l.split() for l in lines[start:] if l.strip()]
    return np.array(rows, dtype=float)


def _parse_re_theta(path):
    with open(path) as f:
        for line in f:
            m = _RE_THETA_RE.search(line)
            if m:
                return float(m.group(1))
    return None


class RefDataset:
    # u: velocity/RMS profile table, or None if the source didn't ship one.
    # budgets: component name ('uu','vv','ww','uv','k') -> budget table, only
    # for the components that are actually available.
    def __init__(self, source, Re_theta, u=None, budgets=None):
        self.source = source
        self.Re_theta = Re_theta
        self.u = u
        self.budgets = budgets or {}

    def label(self):
        return rf'{self.source} $Re_\theta$ = {self.Re_theta:.0f}'


def _closest_dataset(datasets, target_Re_theta):
    if not datasets:
        return None
    return min(datasets, key=lambda d: abs(d.Re_theta - target_Re_theta))


def _load_schlatter_datasets(top_dir):
    datasets = []
    for folder in sorted(glob.glob(os.path.join(top_dir, 'schlatter_ref_*'))):
        budgets = {}
        for comp in ('uu', 'vv', 'ww', 'uv', 'k'):
            matches = sorted(glob.glob(os.path.join(folder, f'bud_*_dns_{comp}.prof*')))
            if matches:
                budgets[comp] = _read_profile_table(matches[0])
        if not budgets:
            continue
        vel_matches = sorted(glob.glob(os.path.join(folder, 'vel_*_dns.prof*')))
        u = _read_profile_table(vel_matches[0]) if vel_matches else None
        header_file = vel_matches[0] if vel_matches else sorted(glob.glob(os.path.join(folder, 'bud_*')))[0]
        re_theta = _parse_re_theta(header_file)
        if re_theta is None:
            continue
        datasets.append(RefDataset('Schlatter', re_theta, u=u, budgets=budgets))
    return sorted(datasets, key=lambda d: d.Re_theta)


def _load_jimenez_datasets(top_dir):
    datasets = []
    for folder in sorted(glob.glob(os.path.join(top_dir, 'jimenez_ref*'))):
        for prof_file in sorted(glob.glob(os.path.join(folder, 'Re_theta.*.prof.txt'))):
            m = re.search(r'Re_theta\.([\d.]+)\.prof\.txt$', prof_file)
            if not m:
                continue
            tag = m.group(1)
            budgets = {}
            for comp in ('uu', 'vv', 'ww', 'uv'):
                matches = glob.glob(os.path.join(folder, f'Re_theta.{tag}.bal.{comp}.txt'))
                if matches:
                    budgets[comp] = _read_profile_table(matches[0])
            u = _read_profile_table(prof_file)
            re_theta = _parse_re_theta(prof_file) or float(tag)
            datasets.append(RefDataset('Jimenez', re_theta, u=u, budgets=budgets))
    return sorted(datasets, key=lambda d: d.Re_theta)


schlatter_datasets = _load_schlatter_datasets(top_dir) if add_schlatter_reference else []
jimenez_datasets   = _load_jimenez_datasets(top_dir)   if add_jimenez_reference   else []

# (column, colour-index) pairs used when overlaying literature budgets onto
# the simulation's Production/Convection/.../Residual colour scheme. Schlatter
# has no separate pressure-transport column (it's folded into velp = PRS +
# Pres. Trans.), and neither source reports every term Res_ij tracks.
_SCHLATTER_BUDGET_COLS = [(3, 0), (2, 2), (4, 4), (6, 6), (5, 10), (7, 12), (8, 14)]
_JIMENEZ_BUDGET_COLS   = [(2, 0), (3, 2), (6, 4), (5, 6), (4, 10), (7, 12)]


def _plot_literature_budget(dataset, comp, cols, colors, linestyle):
    table = dataset.budgets.get(comp)
    if table is None:
        return
    yp_ref = table[:, 1]
    for col, c in cols:
        plt.semilogx(yp_ref, table[:, col], color=colors(c), linestyle=linestyle)


def _shades(n, cmap_name, lo=0.4, hi=0.9):
    cmap = plt.get_cmap(cmap_name)
    if n <= 1:
        return [cmap(hi)] * max(n, 0)
    return [cmap(lo + (hi - lo) * i / (n - 1)) for i in range(n)]



def plot_pressure(x, P_meany):
    plt.figure(0, figsize=plot_style.set_size(width_pts=455.24408, fraction=.495))
    plt.plot(x, P_meany[:,0,0])
    # plt.savefig('pressure_orig')
    plot_style.save('pressure_orig')
    plt.close()
    
def _index_at_wall_height(XYZ, y_target):
    """For each streamwise station, return the wall-normal grid index whose
    distance above the (curved) wall is closest to y_target.

    XYZ has shape (Nk, Nj, Nz, 3) with axis 0 = wall-normal (index 0 = wall),
    axis 1 = streamwise. Returns an integer array of length Nj.
    """
    wall = XYZ[0:1, :, 0, :]                      # (1, Nj, 3) wall points
    dist = np.linalg.norm(XYZ[:, :, 0, :] - wall, axis=-1)   # (Nk, Nj)
    return np.argmin(np.abs(dist - y_target), axis=0)        # (Nj,)

def plot_U_over_x(UVW, x, XYZ, y_target):

    k = _index_at_wall_height(XYZ, y_target)
    u = UVW[k, np.arange(UVW.shape[1]), 0, 0]
    plt.figure(1, figsize=plot_style.set_size(width_pts=455.24408, fraction=.495))
    plt.plot(x, u)
    plt.xlabel('streamwise direction')
    plt.ylabel(f'u at wall-normal height y={y_target:g}')
    # plt.savefig('Uoverx_orig')
    plot_style.save('Uoverx_orig')
    plt.close()

def plot_Urot_over_x(UVW, x, XYZ, y_target):

    k = _index_at_wall_height(XYZ, y_target)
    u = UVW[k, np.arange(UVW.shape[1]), 0, 0]
    plt.figure(2, figsize=plot_style.set_size(width_pts=455.24408, fraction=.495))
    plt.plot(x, u)
    plt.xlabel('streamwise direction')
    plt.ylabel(f'u_rot at wall-normal height y={y_target:g}')
    # plt.savefig('Uoverx_rot')
    plot_style.save('Uoverx_rot')
    plt.close()
    

def plot_bernoulli(x, y, UVW, P, rho):
    ix_plot = np.arange(29, len(y), 25, dtype=int) 
    print(ix_plot)
    plt.figure(3, figsize=plot_style.set_size(width_pts=455.24408, fraction=.495))
    for ix in ix_plot:
        bernoulli = P[ix,:,0,0]/rho +1/2 * (UVW[ix,:,0,0]**2 + UVW[ix,:,0,1]**2 + UVW[ix,:,0,2]**2 )
        print('UVW mag shape',UVW[ix,:,0,0].shape )
        lengend_entry=round(y[ix], 2)
        print(lengend_entry)
        plt.plot(x, bernoulli, label=lengend_entry)
    
    plt.ylim([0,0.9])    
    plt.xlabel('streamwise direction')
    plt.ylabel('P/rho+ 1/2 * magU^2')
    plt.legend()
    # plt.savefig('bernoulli.pdf', format = 'pdf')
    plot_style.save('bernoulli')
    plt.close()
    
def plot_free_stream_diagnostic(UVW, y, ix_plot):
     #%% 
    #######################################################################
    plt.figure(4, figsize=plot_style.set_size(width_pts=455.24408, fraction=.495))
    plt.plot(y, UVW[:,ix_plot,0,0] , label='file '+str(kkk))
    plt.xlim([0, np.max(y)  ])
    plt.ylim([.999, 1.001 ])
    plt.xlabel(r'$y/\delta_{99, in}$')
    plt.ylabel(r'$<u>/U_{\infty, in} $')
    plt.title('Free stream diagnostic')
    plt.legend()
    plt.grid(which='both')
    
def _moving_average(a, window):
    """Centered moving average with edge-reflection padding (keeps length)."""
    window = int(window)
    if window < 2:
        return a
    if window % 2 == 0:          # force odd window so it stays centered
        window += 1
    pad = window // 2
    a_pad = np.pad(a, pad, mode='reflect')
    kernel = np.ones(window) / window
    return np.convolve(a_pad, kernel, mode='valid')

def plot_clauser(s, dStar, tauW_over_rho, P_e, smooth_len=0.0, polyorder=3):
    """Clauser pressure-gradient parameter

        beta(s) = dStar(s) / tau_w(s) * dP_e/ds

    tauW_over_rho is tau_w/rho and P_e is the edge (kinematic) pressure p_e/rho,
    so rho cancels and beta is dimensionless. s is the streamwise arc length and
    P_e the edge pressure sampled at each station (same length as s).

    dP_e/ds is obtained with a Savitzky-Golay filter (local polynomial fit,
    differentiated analytically) -- far cleaner than differentiating a noisy
    finite difference. smooth_len is the filter window length in units of s
    (larger = smoother); polyorder is the local polynomial degree. Set
    smooth_len to 0 to fall back to a plain finite difference.
    """
    from scipy.signal import savgol_filter
    # raw (unsmoothed) beta, kept only for visual comparison
    beta = dStar / tauW_over_rho * np.gradient(P_e, s)
    ds_mean = np.mean(np.diff(s))
    if smooth_len and smooth_len > 0:
        win = int(round(smooth_len / ds_mean))          # window in points
        win = max(win, polyorder + 2)
        if win % 2 == 0:                                 # savgol needs odd window
            win += 1
        dPe_ds = savgol_filter(P_e, win, polyorder, deriv=1, delta=ds_mean)
        #label  = f'Savitzky-Golay (~{smooth_len:g} in $s$)'
        label='beta(s)'
    else:
        dPe_ds = np.gradient(P_e, s)
        label  = 'finite difference'
    beta_smooth = dStar / tauW_over_rho * dPe_ds
    plt.figure(50, figsize=plot_style.set_size(width_pts=455.24408, fraction=.495))
    #plt.plot(s, beta, color='0.7', lw=0.8, label='raw')
    plt.plot(s, beta_smooth, color='C0', label=label)
    plt.xlabel(r'$s$')
    plt.ylabel(r'$\beta$')
    # show only up to s = 550 and scale y to the smoothed data within that window
    s_max = 800
    mask  = s <= s_max
    plt.xlim(0, s_max)
    bmin, bmax = beta_smooth[mask].min(), beta_smooth[mask].max()
    pad = 0.05 * (bmax - bmin) if bmax > bmin else 0.1
    #plt.ylim(bmin - 20*pad, bmax + 20*pad)
    plt.ylim(-1,1)
    plt.title('Clauser pressure-gradient parameter')
    plt.legend()
    # plt.savefig('clauser_beta.png', dpi=200)
    # plt.savefig('clauser_beta.pdf', format='pdf')
    plot_style.save('clauser_beta')
    plt.close()
    return beta_smooth

def plot_pressure_over_s(s, P_e, s_marks=None, s_max=None):
    """Raw pressure along the wall arc length s -- no derivative, no dStar/tau_w
    scaling, no smoothing.

    This is the most direct view of the pressure gradient there is: the only thing
    that matters is the CHANGE in P over a stretch of s, because the datum is
    arbitrary (an incompressible solve fixes p only up to a constant, so the level
    differs between runs and must never be compared). Use this to judge whether a
    case accelerates (p falling) or decelerates (p rising) before looking at
    anything derived. beta in particular multiplies dP/ds by dStar/tau_w, which can
    differ substantially between two cases and makes their beta curves look more
    alike (or less) than their actual pressure gradients are.

    s_marks: optional list of s values to mark with vertical lines (e.g. the arc
    entry/exit, where the curvature jumps and every s-derivative spikes).
    """
    plt.figure(51, figsize=plot_style.set_size(width_pts=455.24408, fraction=.495))
    plt.plot(s, P_e, color='C0')
    for sm in (s_marks or []):
        plt.axvline(sm, color='0.7', lw=0.8, ls=':')
    plt.xlabel(r'$s$')
    plt.ylabel(r'$p_w$')
    if s_max:
        plt.xlim(0, s_max)
    plt.title('Wall pressure along the arc length')
    plot_style.save('pressure_over_s')
    plt.close()


def plot_dpds_over_s(s, P_e, smooth_len=0.0, polyorder=3, s_marks=None, s_max=None):
    """dP/ds along the wall arc length s -- beta without the dStar/tau_w prefactor.

    Same Savitzky-Golay option as plot_clauser(): smooth_len is the window in units
    of s, 0 (default) falls back to a plain np.gradient, matching what
    write_beta_dpds() puts in the txt file. If the case has a short-wavelength
    ripple in p, differentiating amplifies it (a derivative multiplies amplitude by
    2*pi/lambda), so a small wiggle in p can dominate dP/ds -- set smooth_len to a
    few times the ripple wavelength to see the underlying trend instead.
    """
    from scipy.signal import savgol_filter
    ds_mean = np.mean(np.diff(s))
    if smooth_len and smooth_len > 0:
        win = int(round(smooth_len / ds_mean))
        win = max(win, polyorder + 2)
        if win % 2 == 0:
            win += 1
        dPds  = savgol_filter(P_e, win, polyorder, deriv=1, delta=ds_mean)
        label = rf'Savitzky-Golay (${smooth_len:g}$ in $s$)'
    else:
        dPds  = np.gradient(P_e, s)
        label = 'finite difference'
    plt.figure(52, figsize=plot_style.set_size(width_pts=455.24408, fraction=.495))
    plt.plot(s, dPds, color='C0', label=label)
    plt.axhline(0.0, color='0.5', lw=0.8)
    for sm in (s_marks or []):
        plt.axvline(sm, color='0.7', lw=0.8, ls=':')
    plt.xlabel(r'$s$')
    plt.ylabel(r'$\mathrm{d}p_w/\mathrm{d}s$')
    if s_max:
        plt.xlim(0, s_max)
    plt.title('Wall-pressure gradient along the arc length')
    plt.legend()
    plot_style.save('dpds_over_s')
    plt.close()
    return dPds


def _anchor_index(s, s_anchor):
    """Index of the station closest to s_anchor (or 0 if s_anchor is None)."""
    return 0 if s_anchor is None else int(np.argmin(np.abs(np.asarray(s) - s_anchor)))


def plot_delta_pressure_over_s(s, P_w, s_anchor=None, s_marks=None, s_max=None):
    """Wall pressure RELATIVE to its value at s_anchor: p_w(s) - p_w(s_anchor).

    This is the plot to use when comparing two cases. An incompressible solve fixes
    p only up to an additive constant, so the absolute level carries no information
    and differs arbitrarily between runs -- subtracting the value at a common anchor
    (e.g. the start of the bend) removes it and leaves only the physically
    meaningful part: how far the pressure has risen or fallen since the anchor.

    Unlike beta this has no dStar/tau_w prefactor, so it shows the imposed pressure
    gradient without the boundary layer's own response mixed in.
    """
    i0 = _anchor_index(s, s_anchor)
    plt.figure(53, figsize=plot_style.set_size(width_pts=455.24408, fraction=.495))
    plt.plot(s, np.asarray(P_w) - P_w[i0], color='C0')
    plt.axhline(0.0, color='0.5', lw=0.8)
    for sm in (s_marks or []):
        plt.axvline(sm, color='0.7', lw=0.8, ls=':')
    plt.xlabel(r'$s$')
    plt.ylabel(rf'$p_w(s) - p_w({s[i0]:.0f})$')
    if s_max:
        plt.xlim(0, s_max)
    plt.title('Wall-pressure change relative to the anchor station')
    plot_style.save('delta_pressure_over_s')
    plt.close()


def plot_Ue_over_s(s, Uinf, s_anchor=None, s_marks=None, s_max=None):
    """Edge velocity normalised by its value at s_anchor, U_e(s)/U_e(s_anchor).

    The most direct statement of "is this a ZPG": a true zero-pressure-gradient
    boundary layer has a constant edge velocity, so this curve should be flat at 1.
    Needs no explanation of what beta means, and (like plot_delta_pressure_over_s)
    carries no dStar/tau_w prefactor.
    """
    i0 = _anchor_index(s, s_anchor)
    plt.figure(54, figsize=plot_style.set_size(width_pts=455.24408, fraction=.495))
    plt.plot(s, np.asarray(Uinf) / Uinf[i0], color='C0')
    plt.axhline(1.0, color='0.5', lw=0.8)
    for sm in (s_marks or []):
        plt.axvline(sm, color='0.7', lw=0.8, ls=':')
    plt.xlabel(r'$s$')
    plt.ylabel(rf'$U_e(s)/U_e({s[i0]:.0f})$')
    if s_max:
        plt.xlim(0, s_max)
    plt.title('Edge velocity (flat = ZPG)')
    plot_style.save('Ue_over_s')
    plt.close()


def plot_accel_param_over_s(s, Uinf, fluid_nu, smooth_len=0.0, polyorder=3,
                            s_marks=None, s_max=None):
    """Acceleration parameter K = (nu/U_e^2) dU_e/ds along the arc length.

    Standard in the TBL literature for exactly the "how far from ZPG am I" question
    (K > 0 accelerating/favourable, K < 0 decelerating/adverse; |K| ~ 3e-6 is the
    usual relaminarisation threshold), so a reader recognises it without being
    walked through a definition.

    Like plot_dpds_over_s this is a derivative, so a short-wavelength ripple in U_e
    gets amplified by 2*pi/lambda -- set smooth_len to a few times the ripple
    wavelength to see the trend underneath.
    """
    from scipy.signal import savgol_filter
    s = np.asarray(s, dtype=float); Ue = np.asarray(Uinf, dtype=float)
    ds_mean = np.mean(np.diff(s))
    if smooth_len and smooth_len > 0:
        win = int(round(smooth_len / ds_mean))
        win = max(win, polyorder + 2)
        if win % 2 == 0:
            win += 1
        dUe_ds = savgol_filter(Ue, win, polyorder, deriv=1, delta=ds_mean)
        label  = rf'Savitzky-Golay (${smooth_len:g}$ in $s$)'
    else:
        dUe_ds = np.gradient(Ue, s)
        label  = 'finite difference'
    K = fluid_nu / Ue**2 * dUe_ds
    plt.figure(55, figsize=plot_style.set_size(width_pts=455.24408, fraction=.495))
    plt.plot(s, K * 1e6, color='C0', label=label)
    plt.axhline(0.0, color='0.5', lw=0.8)
    for sm in (s_marks or []):
        plt.axvline(sm, color='0.7', lw=0.8, ls=':')
    plt.xlabel(r'$s$')
    plt.ylabel(r'$K \times 10^{6}$')
    if s_max:
        plt.xlim(0, s_max)
    plt.title('Acceleration parameter (flat at 0 = ZPG)')
    plt.legend()
    plot_style.save('accel_param_over_s')
    plt.close()
    return K


def plot_tbl_thickness(x,delta99, delta99_approx, dStar, dStar_trunc, theta, theta_trunc):
    plt.figure(5, figsize=plot_style.set_size(width_pts=455.24408, fraction=.495))
    plt.plot(x , delta99 , 'r' , label=r'$\delta_{99}$')
    plt.plot(x , delta99_approx , '--r', label=r'$\delta_{99}$ approx.' )
    plt.plot(x , dStar , 'g' , label=r'$\delta^*$' )
    plt.plot(x , dStar_trunc , '--g' , label=r'$\delta^*$ trunc.' )
    plt.plot(x , theta , 'b' , label=r'$\theta$' )
    plt.plot(x , theta_trunc , '--b' , label=r'$\theta$ trunc.' )
    # plt.plot(refdata_X , refdata_RedStar , ':g' , label='Eitel Amor et al' )
    # plt.plot(refdata_X , refdata_ReTheta , '--k' , label='Eitel Amor et al' )
    #plt.ylim([0, 10])
    plt.xlim([0, np.max(x)])
    plt.legend()
    plt.xlabel('s')
    #plt.ylabel(r'$Re_{\theta},\ Re_{\delta_{99}},\ Re_{\delta^*}$')
    # plt.savefig('tbl_thickness.png')
    # plt.savefig('tbl_thickness.pdf', format = 'pdf')
    plot_style.save('tbl_thickness')
    plt.close()
    
def plot_reynolds_numbers(x, Re_dStar, Re_theta):
    plt.figure(6, figsize=plot_style.set_size(width_pts=455.24408, fraction=.495))
    # plt.plot(x , Re_delta99 , 'r' , label=r'$Re_{\delta_{99}}$')
    plt.plot(x , Re_dStar , 'g', label=r'$Re_{\delta^*}$' )
    plt.plot(x , Re_theta , 'b' , label=r'$Re_{\theta}$' )
    #if kkk==1:
     #   plt.plot(refdata_X , refdata_RedStar , '--g' , label=r'$Re_{\delta^*}$ Ref.' )
      #  plt.plot(refdata_X , refdata_ReTheta , '--b' , label=r'$Re_{\theta}$ Ref.' )
        # plt.plot(refdata_X , theta_trunc , '--b' , label=r'$\theta$ trunc.' )
    # plt.plot(refdata_X , refdata_RedStar , ':g' , label='Eitel Amor et al' )
    # plt.plot(refdata_X , refdata_ReTheta , '--k' , label='Eitel Amor et al' )
    # plt.ylim([0, 50])
    plt.xlim([0, np.max(x)])
    # nanmax: stations whose BL exceeds the sampled wall-normal height come back NaN
    # from calculate_TBL_quantities; ignore them here instead of crashing set_ylim.
    ymax = np.nanmax(Re_dStar)
    plt.ylim([0, 1.1*ymax if np.isfinite(ymax) else 1.0])
    plt.legend()
    plt.xlabel(r's')
    # plt.savefig('reynolds_numbers.png')
    # plt.savefig('reynolds_numbers.pdf', format = 'pdf')
    plot_style.save('reynolds_numbers')
    plt.close()
    
def plot_u(y, delta99, UVW, Uinf, ix_plot, plot_spos, plot_Re_theta):
    #ix = 200
    plt.figure(7, figsize=plot_style.set_size(width_pts=455.24408, fraction=.495))
    for i, ix in enumerate(ix_plot):
        plt.plot(y/delta99[ix],UVW[:,ix,0,0]/Uinf[ix], label=rf's = {plot_spos[i]:.0f}, $Re_\theta$ = {plot_Re_theta[i]:.0f}')
    # plt.plot(refdata_yoverd99,refdata_up)
    #plt.plot(y/delta99[ix],UVW[:,ix,0,0]/Uinf[ix])
    plt.xlim([0, 1.2  ])
    plt.ylim([0, 1.05 ])
    plt.xlabel(r'$y/\delta_{99}$')
    plt.ylabel(r'$<u>/U_\infty $')
    plt.yticks(np.arange(0,1.1,.1))
    plt.grid(which='both')
    plt.legend()
    # plt.savefig('u_over_y')
    # plt.savefig('UoverYdelta99.pdf', format = 'pdf')
    plot_style.save('u_over_y')
    plt.close()
    
def plot_near_wall_u(y, delta_nu, UVW, u_tau, ix_plot, plot_spos, plot_Re_theta, ref=None):
    # ref: optional dict for a second own configuration (e.g. TBL_pp_local_script_2configs.py),
    #      plotted dashed. Same keys as plot_budgets' ref, plus 'UVW'; expects the
    #      same target Re_theta stations ('ix_plot') as ix_plot here. Colours for
    #      config1/config2 come from plot_style.PLOT_COLOR_SCHEMES['velocity'].
    plt.figure(8, figsize=plot_style.set_size(width_pts=455.24408, fraction=.495))
    colors1 = plot_style.get_colors('velocity', 'config1')
    colors2 = plot_style.get_colors('velocity', 'config2')
    for i, ix in enumerate(ix_plot):
        plt.semilogx(y/delta_nu[ix],UVW[:,ix,0,0]/u_tau[ix], color=colors1(2*i),
                     label=rf's = {plot_spos[i]:.0f}, $Re_\theta$ = {plot_Re_theta[i]:.0f}')
        if ref is not None:
            ixr = ref['ix_plot'][i]
            plt.semilogx(ref['y']/ref['delta_nu'][ixr], ref['UVW'][:,ixr,0,0]/ref['u_tau'][ixr],
                         color=colors2(2*i), linestyle='--',
                         label=ref.get('label', 'reference config') if i == 0 else None)
    # every available literature Re_theta station is shown here (not just the
    # closest one) so the near-wall collapse can be judged across the range.
    if add_schlatter_reference:
        ds_list = [d for d in schlatter_datasets if d.u is not None]
        for ds, color in zip(ds_list, _shades(len(ds_list), 'Reds')):
            plt.semilogx(ds.u[:, 1], ds.u[:, 2], color=color, linestyle=':', label=ds.label())
    if add_jimenez_reference:
        ds_list = [d for d in jimenez_datasets if d.u is not None]
        for ds, color in zip(ds_list, _shades(len(ds_list), 'Blues')):
            plt.semilogx(ds.u[:, 1], ds.u[:, 6], color=color, linestyle='-.', label=ds.label())
    plt.semilogx(np.linspace(0,15,20) , np.linspace(0,15,20) , '--k' )
    plt.semilogx(np.linspace(10,10000,10) , 1/.41*np.log(np.linspace(10,10000,10))+5.2 , '--k' )
    if ref is not None:
        plt.title(f"(dashed: {ref.get('label', 'reference config')})")
    plt.xlim([.5, 1000])
    plt.ylim([0,30])
    plt.xlabel('$y^+$')
    plt.ylabel('$U^+$')
    plt.legend()
    # plt.savefig('UplusNearWall')
    # plt.savefig('UplusNearWall.pdf', format = 'pdf')
    plot_style.save('UplusNearWall')
    plt.close()
    
def plot_fluctuations(y, delta_nu, u_tau, Rij, ix_plot, plot_spos, plot_Re_theta, ref=None):
    # ref: optional dict for a second own configuration, plotted dashed. Same
    #      keys as plot_budgets' ref, plus 'Rij'. Colours for config1/config2
    #      come from plot_style.PLOT_COLOR_SCHEMES['fluctuations'].
    plt.figure(9, figsize=plot_style.set_size(width_pts=455.24408, fraction=0.7))
    colors1 = plot_style.get_colors('fluctuations', 'config1')
    colors2 = plot_style.get_colors('fluctuations', 'config2')
    # one colour per Re_theta station (same logic as plot_RMS); all four
    # components (uu, vv, ww, uv) share the station colour, legend labels the
    # station once (on the first component only) to avoid duplicate entries.
    # Literature reference: only the station's closest available Re_theta is
    # overlaid (dotted = Schlatter, dash-dot = Jimenez), in the station colour.
    schlatter_by_station = [_closest_dataset(schlatter_datasets, rt) if add_schlatter_reference else None
                             for rt in plot_Re_theta]
    jimenez_by_station   = [_closest_dataset(jimenez_datasets, rt) if add_jimenez_reference else None
                             for rt in plot_Re_theta]
    for k in range(4):
        for i, ix in enumerate(ix_plot):
            label = rf's = {plot_spos[i]:.0f}, $Re_\theta$ = {plot_Re_theta[i]:.0f}' if k == 0 else None
            plt.semilogx(y/delta_nu[ix], Rij[:,ix,0,k]/u_tau[ix]**2, color=colors1(2*i), label=label)
            if ref is not None:
                ixr = ref['ix_plot'][i]
                ref_label = ref.get('label', 'reference config') if (k == 0 and i == 0) else None
                plt.semilogx(ref['y']/ref['delta_nu'][ixr], ref['Rij'][:,ixr,0,k]/ref['u_tau'][ixr]**2,
                             color=colors2(2*i), linestyle='--', label=ref_label)
            sd = schlatter_by_station[i]
            if sd is not None and sd.u is not None:
                val = sd.u[:, 3 + k]**2 if k < 3 else sd.u[:, 6]
                plt.semilogx(sd.u[:, 1], val, color=colors1(2*i), linestyle=':')
            jd = jimenez_by_station[i]
            if jd is not None and jd.u is not None:
                val = jd.u[:, 2 + k]**2 if k < 3 else jd.u[:, 5]
                plt.semilogx(jd.u[:, 1], val, color=colors1(2*i), linestyle='-.')
    if ref is not None or add_schlatter_reference or add_jimenez_reference:
        note = ', '.join(filter(None, [
            f"dashed: {ref.get('label', 'reference config')}" if ref is not None else None,
            'dotted: Schlatter closest $Re_\\theta$' if add_schlatter_reference else None,
            'dash-dot: Jimenez closest $Re_\\theta$' if add_jimenez_reference else None,
        ]))
        plt.title(note)
   # plt.xlim([.5, 1000])
    #plt.ylim([-3, 10])
    plt.xlabel(r'$y^{+}$')
    plt.ylabel(r'$\langle u_i u_j \rangle^+$')
    plt.legend()
    # plt.savefig('fluctuations')
    # plt.savefig('fluctuations.pdf', format = 'pdf')
    plot_style.save('fluctuations')
    plt.close()

def plot_RMS(y, delta_nu, u_tau, Rij, ix_plot, plot_spos, plot_Re_theta, ref=None):
    # ref: optional dict for a second own configuration, plotted dashed. Same
    #      keys as plot_budgets' ref, plus 'Rij'. Colours for config1/config2
    #      come from plot_style.PLOT_COLOR_SCHEMES['rms'].
    plt.figure(5, figsize=plot_style.set_size(width_pts=455.24408, fraction=0.7))
    colors1 = plot_style.get_colors('rms', 'config1')
    colors2 = plot_style.get_colors('rms', 'config2')
    # every available literature Re_theta station is shown here (not just the
    # closest one) so the RMS collapse can be judged across the range.
    schlatter_u = [d for d in schlatter_datasets if d.u is not None] if add_schlatter_reference else []
    jimenez_u   = [d for d in jimenez_datasets if d.u is not None]   if add_jimenez_reference   else []
    schlatter_colors = _shades(len(schlatter_u), 'Reds')
    jimenez_colors   = _shades(len(jimenez_u), 'Blues')
    for k in range(3):
        for i, ix in enumerate(ix_plot):
            label = rf's = {plot_spos[i]:.0f}, $Re_\theta$ = {plot_Re_theta[i]:.0f}' if k == 0 else None
            plt.semilogx(y/delta_nu[ix],np.sqrt(Rij[:,ix,0,k]/u_tau[ix]**2), color = colors1(2*i), label = label)
            if ref is not None:
                ixr = ref['ix_plot'][i]
                ref_label = ref.get('label', 'reference config') if (k == 0 and i == 0) else None
                plt.semilogx(ref['y']/ref['delta_nu'][ixr], np.sqrt(ref['Rij'][:,ixr,0,k]/ref['u_tau'][ixr]**2),
                             color=colors2(2*i), linestyle='--', label=ref_label)
        for ds, color in zip(schlatter_u, schlatter_colors):
            plt.semilogx(ds.u[:, 1], ds.u[:, 3 + k], color=color, linestyle=':',
                         label=ds.label() if k == 0 else None)
        for ds, color in zip(jimenez_u, jimenez_colors):
            plt.semilogx(ds.u[:, 1], ds.u[:, 2 + k], color=color, linestyle='-.',
                         label=ds.label() if k == 0 else None)

    if ref is not None:
        plt.title(f"(dashed: {ref.get('label', 'reference config')})")
    plt.xlabel(r'$y^{+}$')
    plt.ylabel(r'$RMS $')
    plt.legend()
    plt.xlim([.5, 1000])
    plt.ylim([0, 3.5])
    # plt.savefig('RMS.pdf', format = 'pdf')
    plot_style.save('RMS')
    plt.close()
    
def plot_Re_theta(Re_theta, tauW_over_rho, Uinf):
    plt.figure(10, figsize=plot_style.set_size(width_pts=455.24408, fraction=.495))
    plt.plot(Re_theta , 2*tauW_over_rho/Uinf**2 , label='file '+str(kkk))
    if kkk==1:
        plt.plot(Re_theta , .024 * Re_theta**(-0.25) , ':g' , label='Smits et al')
        plt.plot(Re_theta , 2 / (1/.384*np.log(Re_theta)+4.127  )**2 , ':r' , label='Coles–Fernholz')
        #plt.plot(refdata_ReTheta , refdata_cf , '--k' , label='Ref.')
    plt.xlim([0, np.max(Re_theta)])
    plt.xlabel(r'$Re_{\theta}$')
    plt.ylabel(r'$c_f$')
    plt.legend()
    
def plot_ReTheta_ReTau(Re_theta, Re_tau):
    plt.figure(11, figsize=plot_style.set_size(width_pts=455.24408, fraction=.495))
    plt.plot(Re_theta , Re_tau  )
    # if kkk==0:
        # plt.plot(refdata_ReTheta,refdata_RedStar)
    plt.xlabel(r'$Re_{\theta}$')
    plt.ylabel(r'$Re_{\tau}$')
    plt.legend()
    # plt.show()
    
def plot_budgets(plot_spos, names,  ix_plot, y, delta_nu, u_tau, Prod_ij, Conv_ij, Diss_ij,PRS_ij, PressTrans_ij , TurbTrans_ij, ViscDiff_ij, Res_ij, plot_Re_theta, ref=None):
    # ref: optional dict with a reference (e.g. flat-plate/config2) dataset, plotted
    #      dashed. expected keys: 'y', 'delta_nu', 'u_tau', 'ix_plot' (indices matching
    #      the same Re_theta stations) and the 8 budget tensors below. Colours for
    #      config1/config2 come from plot_style.PLOT_COLOR_SCHEMES['budgets'].
    colors  = plot_style.get_colors('budgets', 'config1')
    colors2 = plot_style.get_colors('budgets', 'config2')

    # (label, tensor, colour index) — single source of truth for both cases
    terms = [('Production',   Prod_ij,        0),
             ('Convection',   Conv_ij,        2),
             ('Dissipation',  Diss_ij,        4),
             ('PRS',          PRS_ij,         6),
             ('Pres. Trans.', PressTrans_ij,  8),
             ('Turb. Tran.',  TurbTrans_ij,  10),
             ('Visc. Diff',   ViscDiff_ij,   12),
             ('Residual',     Res_ij,        14)]
    ref_terms = ['Prod_ij','Conv_ij','Diss_ij','PRS_ij',
                 'PressTrans_ij','TurbTrans_ij','ViscDiff_ij','Res_ij']

    comp_names = ['uu', 'vv', 'ww', 'uv']
    for x, ix in enumerate(ix_plot): # range(1000,Nxyz[1],200):
        # closest available literature Re_theta to *this* station, so a
        # multi-station figure set naturally picks a different reference per
        # station once several Schlatter/Jimenez Re_theta folders exist.
        schlatter_ds = _closest_dataset(schlatter_datasets, plot_Re_theta[x]) if add_schlatter_reference else None
        jimenez_ds   = _closest_dataset(jimenez_datasets, plot_Re_theta[x])   if add_jimenez_reference   else None
        for k in range(4):
            plt.figure(12+x + k, figsize=plot_style.set_size(width_pts=455.24408, fraction=0.8))
            title = rf's = {plot_spos[x]:.0f}, $Re_\theta$ = {plot_Re_theta[x]:.0f}'
            if ref is not None:
                title += f"  (dashed: {ref.get('label', 'reference config')})"
            if schlatter_ds is not None:
                title += f'  (dotted: {schlatter_ds.label()})'
            if jimenez_ds is not None:
                title += f'  (dash-dot: {jimenez_ds.label()})'
            plt.title(title)
            scale = u_tau[ix]**3/delta_nu[ix]
            yp    = y/delta_nu[ix]
            for label, tensor, c in terms:
                plt.semilogx(yp, tensor[:,ix,0,k]/scale, color=colors(c), label=label)
            if ref is not None:
                ixr   = ref['ix_plot'][x]
                scr   = ref['u_tau'][ixr]**3/ref['delta_nu'][ixr]
                ypr   = ref['y']/ref['delta_nu'][ixr]
                for (label, _, c), key in zip(terms, ref_terms):
                    plt.semilogx(ypr, ref[key][:,ixr,0,k]/scr, color=colors2(c), linestyle='--')
            # literature DNS reference for this Reynolds-stress component (already
            # in wall units, so plotted directly vs its own y+ = column 1). Its
            # pressure column is the combined velocity-pressure-gradient term
            # (PRS + Pres. Trans.), drawn in the PRS colour. Same term colours as
            # the simulation: dotted for Schlatter, dash-dot for Jimenez.
            comp = comp_names[k]
            if schlatter_ds is not None:
                _plot_literature_budget(schlatter_ds, comp, _SCHLATTER_BUDGET_COLS, colors, ':')
            if jimenez_ds is not None:
                _plot_literature_budget(jimenez_ds, comp, _JIMENEZ_BUDGET_COLS, colors, '-.')
            plt.xlim((.5,1000))
            plt.xlabel(r'$y^{+}$')
            plt.ylabel(names[k])
            # plt.xlim((0, 5))
            plt.legend([label for label, _, _ in terms])
            name = f'budget_Re{plot_Re_theta[x]:.0f}_{names[k]}'
            # plt.savefig(name, dpi = 200, format = 'png')
            # plt.savefig(name+'.pdf', format = 'pdf')
            plot_style.save(name)
            plt.close()

def plot_k_budgets(Nxyz,names,  ix_plot, y, delta_nu, u_tau, Prod_ij, Conv_ij, Diss_ij,PRS_ij, PressTrans_ij , TurbTrans_ij, ViscDiff_ij, Res_ij, plot_Re_theta=None, ref=None):
    # ref: optional dict with a reference (e.g. flat-plate/config2) dataset, plotted
    #      dashed. expected keys as in plot_budgets ('y','delta_nu','u_tau','ix_plot' +
    #      budget tensors). Colours come from plot_style.PLOT_COLOR_SCHEMES['budgets'].
    colors  = plot_style.get_colors('budgets', 'config1')
    colors2 = plot_style.get_colors('budgets', 'config2')

    # TKE budget = 0.5*(uu + vv + ww) contribution of each term
    def tke_term(tensor, ix):
        return 0.5 * np.sum(tensor[:, ix, 0, 0:3], axis=-1)

    terms = [('Production',   Prod_ij,        0),
             ('Convection',   Conv_ij,        2),
             ('Dissipation',  Diss_ij,        4),
             ('PRS',          PRS_ij,         6),
             ('Pres. Trans.', PressTrans_ij,  8),
             ('Turb. Tran.',  TurbTrans_ij,  10),
             ('Visc. Diff',   ViscDiff_ij,   12),
             ('Residual',     Res_ij,        14)]
    ref_terms = ['Prod_ij','Conv_ij','Diss_ij','PRS_ij',
                 'PressTrans_ij','TurbTrans_ij','ViscDiff_ij','Res_ij']

    for xi, ix in enumerate(ix_plot): # range(1000,Nxyz[1],200):
        plt.figure(15 + xi, figsize=plot_style.set_size(width_pts=455.24408, fraction=0.8))
        scale = u_tau[ix]**3/delta_nu[ix]
        yp    = y/delta_nu[ix]
        for label, tensor, c in terms:
            plt.semilogx(yp, tke_term(tensor, ix)/scale, color=colors(c), label=label)
        if ref is not None:
            ixr = ref['ix_plot'][xi]
            scr = ref['u_tau'][ixr]**3/ref['delta_nu'][ixr]
            ypr = ref['y']/ref['delta_nu'][ixr]
            for (label, _, c), key in zip(terms, ref_terms):
                plt.semilogx(ypr, tke_term(ref[key], ixr)/scr, color=colors2(c), linestyle='--')
        # Literature TKE budget (Jimenez provides no k budget, so only Schlatter
        # is available here). Closest available Re_theta to this station (no
        # station Re_theta known -> no reference to compare against).
        schlatter_ds = (_closest_dataset(schlatter_datasets, plot_Re_theta[xi])
                         if add_schlatter_reference and plot_Re_theta is not None else None)
        if schlatter_ds is not None:
            _plot_literature_budget(schlatter_ds, 'k', _SCHLATTER_BUDGET_COLS, colors, ':')

        plt.xlim((0.5,1000))
        plt.xlabel(r'$y^{+}$')
        #plt.ylabel('Energy budgets')
        title = "Energy budgets for k"
        if plot_Re_theta is not None:
            title += rf', $Re_\theta$ = {plot_Re_theta[xi]:.0f}'
        if ref is not None:
            title += f"  (dashed: {ref.get('label', 'reference config')})"
        if schlatter_ds is not None:
            title += f'  (dotted: {schlatter_ds.label()})'
        plt.title(title)
        plt.legend([label for label, _, _ in terms])
        Re_tag = f'_Re{plot_Re_theta[xi]:.0f}' if plot_Re_theta is not None else f'_{xi}'
        # plt.savefig('budget_k'+Re_tag, dpi = 200)
        plot_style.save('budget_k'+Re_tag)
        plt.close()

def plot_premultiplied_bugets(plot_spos,names, colors,  ix_plot, y, delta_nu, u_tau, Prod_ij, Conv_ij, Diss_ij,PRS_ij, PressTrans_ij , TurbTrans_ij, ViscDiff_ij, Res_ij):
    
    ############################################## premultiplicated energy budgets ##################################################
    for x, ix in enumerate(ix_plot): # range(1000,Nxyz[1],200):
        for k in range(4):
            plt.figure(11+x + k, figsize=plot_style.set_size(width_pts=455.24408, fraction=0.8))
            plt.semilogx(y/delta_nu[ix],Prod_ij[:,ix,0,k] /(u_tau[ix]**3/delta_nu[ix]) *  y/delta_nu[ix],color = colors(0))
            plt.semilogx(y/delta_nu[ix],Conv_ij[:,ix,0,k] /(u_tau[ix]**3/delta_nu[ix])  *  y/delta_nu[ix],color = colors(2))
            plt.semilogx(y/delta_nu[ix],Diss_ij[:,ix,0,k] /(u_tau[ix]**3/delta_nu[ix])  *  y/delta_nu[ix],color = colors(4))
            plt.semilogx(y/delta_nu[ix],PRS_ij[:,ix,0,k] /(u_tau[ix]**3/delta_nu[ix]) *  y/delta_nu[ix],color = colors(6))
            plt.semilogx(y/delta_nu[ix],PressTrans_ij[:,ix,0,k]/(u_tau[ix]**3/delta_nu[ix]) *  y/delta_nu[ix],color = colors(8))
            plt.semilogx(y/delta_nu[ix],TurbTrans_ij[:,ix,0,k] /(u_tau[ix]**3/delta_nu[ix]) *  y/delta_nu[ix],color = colors(10))
            plt.semilogx(y/delta_nu[ix],ViscDiff_ij[:,ix,0,k] /(u_tau[ix]**3/delta_nu[ix]) *  y/delta_nu[ix],color = colors(12))
            plt.semilogx(y/delta_nu[ix],Res_ij[:,ix,0,k] /(u_tau[ix]**3/delta_nu[ix]) *  y/delta_nu[ix],color = colors(14))
        
        # if plot_channel_data:
        #     plt.semilogx(yplus_channel,Prod_ij_channel[1:,k] /(u_tau_channel**3/delta_nu_channel)       ,color = colors(0), linestyle = '--')
        #     plt.semilogx(yplus_channel,Diss_ij_channel[1:,k] /(u_tau_channel**3/delta_nu_channel)       ,color = colors(4), linestyle = '--')
        #     plt.semilogx(yplus_channel,PRS_ij_channel[1:,k] /(u_tau_channel**3/delta_nu_channel)        ,color = colors(6), linestyle = '--')
        #     plt.semilogx(yplus_channel,PressTrans_ij_channel[1:,k] /(u_tau_channel**3/delta_nu_channel) ,color = colors(8), linestyle = '--')
        #     plt.semilogx(yplus_channel,TurbTrans_ij_channel[1:,k] /(u_tau_channel**3/delta_nu_channel)  ,color = colors(10), linestyle = '--')
        #     plt.semilogx(yplus_channel,ViscDiff_ij_channel[1:,k] /(u_tau_channel**3/delta_nu_channel)   ,color = colors(12), linestyle = '--')
        #     plt.semilogx(yplus_channel,Res_ij_channel[1:,k] /(u_tau_channel**3/delta_nu_channel)        ,color = colors(14), linestyle = '--')
        
            plt.xlim((.5,1000))
            plt.xlabel(r'$y^{+}$')
            plt.ylabel(names[k])
            # plt.xlim((0, 5))
            plt.legend(('Production', 'Convection','Dissipation','PRS','Pres. Trans.','Turb. Tran.','Visc. Diff','Residual'))
            # plt.savefig('prem_budget_s'+ str(plot_spos[x])+names[k], dpi = 200)
            plot_style.save('prem_budget_s'+ str(plot_spos[x])+names[k])
            plt.close()
    
    
    ix= 1000
    # Energy budgets for TKE k= 1/2 (u'u' + v'v' + w'w')
    Prod_k = 0.5 * np.sum([Prod_ij[:,ix, 0,0], Prod_ij[:,ix,0,1], Prod_ij[:,ix,0,2]], axis = 0)
    Conv_k = 0.5 * np.sum([Conv_ij[:,ix, 0,0], Conv_ij[:,ix,0,1], Conv_ij[:,ix,0,2]], axis = 0)
    Diss_k = 0.5 * np.sum([Diss_ij[:,ix, 0,0], Diss_ij[:,ix,0,1], Diss_ij[:,ix,0,2]], axis = 0)
    PRS_k = 0.5 * np.sum([PRS_ij[:,ix, 0,0], PRS_ij[:,ix,0,1], PRS_ij[:,ix,0,2]], axis = 0)
    PressTrans_k = 0.5 * np.sum([PressTrans_ij[:,ix, 0,0], PressTrans_ij[:,ix,0,1], PressTrans_ij[:,ix,0,2]], axis = 0)
    TurbTrans_k = 0.5 * np.sum([TurbTrans_ij[:,ix, 0,0], TurbTrans_ij[:,ix,0,1], TurbTrans_ij[:,ix,0,2]], axis = 0)
    ViscDiff_k = 0.5 * np.sum([ViscDiff_ij[:,ix, 0,0], ViscDiff_ij[:,ix,0,1], ViscDiff_ij[:,ix,0,2]], axis = 0)
    Res_k = 0.5 * np.sum([Res_ij[:,ix, 0,0], Res_ij[:,ix,0,1], Res_ij[:,ix,0,2]], axis = 0)
    
    print('Prodk shape:', Prod_k.shape)
    print('Prodij shape:', Prod_ij.shape)
    
    plt.figure(20, figsize=plot_style.set_size(width_pts=455.24408, fraction=0.8))
    plt.semilogx(y/delta_nu[ix],Prod_k /(u_tau[ix]**3/delta_nu[ix]) *  y/delta_nu[ix]      ,color = colors(0))
    plt.semilogx(y/delta_nu[ix],Conv_k /(u_tau[ix]**3/delta_nu[ix]) *  y/delta_nu[ix]       ,color = colors(2))
    plt.semilogx(y/delta_nu[ix],Diss_k /(u_tau[ix]**3/delta_nu[ix])*  y/delta_nu[ix]      ,color = colors(4))
    plt.semilogx(y/delta_nu[ix],PRS_k /(u_tau[ix]**3/delta_nu[ix]) *  y/delta_nu[ix]       ,color = colors(6))
    plt.semilogx(y/delta_nu[ix],PressTrans_k /(u_tau[ix]**3/delta_nu[ix]) *  y/delta_nu[ix] ,color = colors(8))
    plt.semilogx(y/delta_nu[ix],TurbTrans_k/(u_tau[ix]**3/delta_nu[ix]) *  y/delta_nu[ix] ,color = colors(10))
    plt.semilogx(y/delta_nu[ix],ViscDiff_k /(u_tau[ix]**3/delta_nu[ix])*  y/delta_nu[ix]    ,color = colors(12))
    plt.semilogx(y/delta_nu[ix],Res_k /(u_tau[ix]**3/delta_nu[ix]) *  y/delta_nu[ix]        ,color = colors(14))
    
    plt.xlim((0.5,1000))
    plt.ylim((-6, 6))
    plt.xlabel(r'$y^{+}$')
    #plt.ylabel('Energy budgets')
    plt.title("Energy budgets for k")
    plt.legend(('Production','Convection','Dissipation','PRS','Pres. Trans.','Turb. Tran.','Visc. Diff','Residual'))
    # plt.savefig('prem_budget_k', dpi = 200)
    plot_style.save('prem_budget_k')
    plt.close()
    
def read_budgets(input_filename):
        #%% Plot parameters
    
    """     #%% load reference data
    refdata_tmp = os.path.expanduser(fnames_refdata_cf)
    df_cf = pd.read_csv(refdata_tmp, sep='\s+', header=None, comment="%",encoding='latin-1') 
    
    # refdata_tmp = os.path.expanduser(fnames_refdata_Re)
    # df_Re = pd.read_csv(refdata_tmp, sep='\s+', header=None, comment="%",encoding='latin-1') 
    refdata_tmp = os.path.expanduser(fnames_refdata_delta)
    df_delta = pd.read_csv(refdata_tmp, sep='\s+', header=None, comment="%",encoding='latin-1') 

    refdata_tmp = os.path.expanduser(fnames_refdata_vel)
    df_vel = pd.read_csv(refdata_tmp, sep='\s+', header=None, comment="%",encoding='latin-1') 

    refdata_X       = df_cf.iloc[:,0]
    refdata_cf      = df_cf.iloc[:,1]
    refdata_dStar   = df_delta.iloc[:,1]
    refdata_theta   = df_delta.iloc[:,2]
    refdata_RedStar = 450*refdata_dStar
    refdata_ReTheta = 450*refdata_theta
    # refdata_RedStar = df_Re.iloc[:,1]
    # refdata_ReTheta = df_Re.iloc[:,2]
    refdata_icut        = np.argmax(refdata_ReTheta)
    refdata_ReTheta     = refdata_ReTheta[:refdata_icut]
    refdata_RedStar     = refdata_RedStar[:refdata_icut]
    refdata_cf          = refdata_cf[:refdata_icut]
    refdata_X           = refdata_X[:refdata_icut]

    refdata_yoverd99 = df_vel.iloc[:,0]
    refdata_yp = df_vel.iloc[:,1]
    refdata_up = df_vel.iloc[:,2]
    refdata_RS = np.array( df_vel.iloc[:,3:7] )
    refdata_RS[:,0:3] = refdata_RS[:,0:3]**2 """

    #%% find x location for line plots based on the given Re_theta
    ix_plot = np.zeros(np.size(plot_ReThetas),dtype=np.int32)
    k=0
    for tmp_retheta in plot_ReThetas:
        ix_plot[k] = np.argmin( np.abs(Re_theta-tmp_retheta) )
        k+=1

    #%% 
    #######################################################################
    plt.figure(0)
    plt.plot(y, UVW[:,ix_plot,0,0] , label='file '+str(kkk))
    plt.xlim([0, np.max(y)  ])
    plt.ylim([.999, 1.001 ])
    plt.xlabel(r'$y/\delta_{99, in}$')
    plt.ylabel(r'$<u>/U_{\infty, in} $')
    plt.title('Free stream diagnostic')
    plt.legend()
    plt.grid(which='both')
    
    #######################################################################
    plt.figure(1)
    plt.plot(x , delta99 , 'r' , label=r'$\delta_{99}$')
    plt.plot(x , delta99_approx , '--r', label=r'$\delta_{99}$ approx.' )
    plt.plot(x , dStar , 'g' , label=r'$\delta^*$' )
    plt.plot(x , dStar_trunc , '--g' , label=r'$\delta^*$ trunc.' )
    plt.plot(x , theta , 'b' , label=r'$\theta$' )
    plt.plot(x , theta_trunc , '--b' , label=r'$\theta$ trunc.' )
    # plt.plot(refdata_X , refdata_RedStar , ':g' , label='Eitel Amor et al' )
    # plt.plot(refdata_X , refdata_ReTheta , '--k' , label='Eitel Amor et al' )
    plt.ylim([0, 50])
    plt.xlim([0, np.max(x)])
    plt.legend()
    
    #######################################################################
    plt.figure(2)
    # plt.plot(x , Re_delta99 , 'r' , label=r'$Re_{\delta_{99}}$')
    plt.plot(x , Re_dStar , 'g', label=r'$Re_{\delta^*}$' )
    plt.plot(x , Re_theta , 'b' , label=r'$Re_{\theta}$' )
    if kkk==1:
        plt.plot(refdata_X , refdata_RedStar , '--g' , label=r'$Re_{\delta^*}$ Ref.' )
        plt.plot(refdata_X , refdata_ReTheta , '--b' , label=r'$Re_{\theta}$ Ref.' )
        # plt.plot(refdata_X , theta_trunc , '--b' , label=r'$\theta$ trunc.' )
    # plt.plot(refdata_X , refdata_RedStar , ':g' , label='Eitel Amor et al' )
    # plt.plot(refdata_X , refdata_ReTheta , '--k' , label='Eitel Amor et al' )
    # plt.ylim([0, 50])
    plt.xlim([0, np.max(x)])
    plt.ylim([0, 1.1*np.max(Re_dStar)])
    plt.legend()

    #######################################################################
    plt.figure(3)
    for ix in ix_plot: 
        plt.plot(y/delta99[ix],UVW[:,ix,0,0]/Uinf[ix])
    # plt.plot(refdata_yoverd99,refdata_up)
    plt.xlim([0, 1.2  ])
    plt.ylim([0, 1.05 ])
    plt.xlabel(r'$y/\delta_{99}$')
    plt.ylabel(r'$<u>/U_\infty $')
    plt.yticks(np.arange(0,1.1,.1))
    plt.grid(which='both')

    #######################################################################
    plt.figure(4)
    for ix in ix_plot: 
        plt.semilogx(y/delta_nu[ix],UVW[:,ix,0,0]/u_tau[ix])
    plt.semilogx(refdata_yp,refdata_up,'--k')
    plt.semilogx(np.linspace(0,15,20) , np.linspace(0,15,20) , '--k' )
    plt.semilogx(np.linspace(10,10000,10) , 1/.41*np.log(np.linspace(10,10000,10))+5.2 , '--k' )
    plt.xlim([.5, 2000])
    plt.ylim([0,30])

    #######################################################################
    plt.figure(5)
    for ix in ix_plot: # range(1000,Nxyz[1],200):
        plt.semilogx(y/delta_nu[ix],Rij[:,ix,0,0:4]/u_tau[ix]**2)
    plt.semilogx(refdata_yp,refdata_RS,'--k')
    plt.xlim([.5, 1000])

    
    #######################################################################
    plt.figure(11)
    plt.plot(Re_theta , 2*tauW_over_rho/Uinf**2 , label='file '+str(kkk))
    if kkk==1:
        plt.plot(Re_theta , .024 * Re_theta**(-0.25) , ':g' , label='Smits et al')
        plt.plot(Re_theta , 2 / (1/.384*np.log(Re_theta)+4.127  )**2 , ':r' , label='Coles–Fernholz')
        plt.plot(refdata_ReTheta , refdata_cf , '--k' , label='Ref.')
    plt.xlim([0, np.max(Re_theta)])
    plt.legend()

    #######################################################################
    plt.figure(12)
    plt.plot(Re_theta , Re_tau  )
    # if kkk==0:
        # plt.plot(refdata_ReTheta,refdata_RedStar)
    plt.xlabel(r'$Re_{\theta}$')
    plt.ylabel(r'$Re_{\tau}$')
    plt.legend()
    # plt.show()

    plt.figure(15)
    plt.plot(x,UVW[-1,:,0,1]/Uinf , label='file '+str(kkk))
    plt.ylabel(r'$v/U_\infty$')
    plt.xlabel(r'$x/\delta_{99,in}$')
    plt.legend()

    plt.figure(100)
    plt.plot(Re_theta , H12 , label='file '+str(kkk) )
    if kkk==1:
        plt.plot(refdata_ReTheta , refdata_RedStar/refdata_ReTheta , '--k' , label='Ref.' )
    plt.ylabel('$H_{12}$')
    plt.xlabel(r'$Re_{\theta}$')
    plt.legend()
    plt.ylim([1.3, 1.7])    
    plt.xlim([np.min(Re_theta) , np.max(Re_theta)])
    
    #%%
    suptitles = [ "u'u'" , "v'v'" , "w'w'" , "u'v'" ]
    if IF_plot_budgets_on_top:
        plt.figure(1001)
    else:
        plt.figure(1000+kkk)
    for ix in ix_plot: # range(1000,Nxyz[1],200):
        for k in range(4):
            plt.subplot(221+k)
            plt.semilogx( y/delta_nu[ix] , Prod_ij[:,ix,0,k]        /(u_tau[ix]**3/delta_nu[ix])    ,'k' )
            plt.semilogx( y/delta_nu[ix] , Diss_ij[:,ix,0,k]        /(u_tau[ix]**3/delta_nu[ix])    ,'b')
            if IF_plot_separate_press_buds:
                plt.semilogx( y/delta_nu[ix] , PRS_ij[:,ix,0,k]         /(u_tau[ix]**3/delta_nu[ix])    ,'r',linewidth=2)
                plt.semilogx( y/delta_nu[ix] , PressTrans_ij[:,ix,0,k]  /(u_tau[ix]**3/delta_nu[ix])    ,'g',linewidth=2)
            else:
                plt.semilogx( y/delta_nu[ix] , VelPressGrad_ij[:,ix,0,k]/(u_tau[ix]**3/delta_nu[ix])    ,'y',linewidth=2)
            plt.semilogx( y/delta_nu[ix] , TurbTrans_ij[:,ix,0,k]   /(u_tau[ix]**3/delta_nu[ix])    ,'c')
            plt.semilogx( y/delta_nu[ix] , ViscDiff_ij[:,ix,0,k]    /(u_tau[ix]**3/delta_nu[ix])    ,'grey')
            plt.semilogx( y/delta_nu[ix] , Conv_ij[:,ix,0,k]        /(u_tau[ix]**3/delta_nu[ix])    ,'orange',linewidth=2)
            plt.semilogx( y/delta_nu[ix] , Res_ij[:,ix,0,k]         /(u_tau[ix]**3/delta_nu[ix])    ,':k')

            refdata = os.path.expanduser(fnames_refdata_buds[k])
            df = pd.read_csv(refdata, sep='\s+', header=None, comment="%") 
            plt.semilogx(df.iloc[:,1],df.iloc[:,2:-1]  ,'--k')

            plt.xlim([ .5 , Re_tau[ix] ])
            plt.title(suptitles[k])

            if IF_plot_separate_press_buds:
                plt.legend(('Production','Dissipation','PRS','Pres. Trans.','Turb. Tran.','Visc. Diff','Conv','Residual'))
            else:
                plt.legend(('Production','Dissipation','Pres. Sum' , 'Turb. Tran.','Visc. Diff','Conv','Residual'))

    #%% Pope style figures
    if IF_plot_budgets_in_Pope_style:
        # figures 7.35-38 (a)
        suptitles = [ "u'u'" , "v'v'" , "w'w'" , "u'v'" ]
        plt.figure(1501)
        for ix in ix_plot: # range(1000,Nxyz[1],200):
            for k in range(4):
                plt.subplot(221+k)
                plt.plot( y/delta99[ix] , Prod_ij[:,ix,0,k]        /Pope_normFactor_ij[:,ix,0,k]    ,'k')
                plt.plot( y/delta99[ix] , Diss_ij[:,ix,0,k]        /Pope_normFactor_ij[:,ix,0,k]    ,'b')
                plt.plot( y/delta99[ix] , PRS_ij[:,ix,0,k]         /Pope_normFactor_ij[:,ix,0,k]    ,'r',linewidth=1)
                plt.plot( y/delta99[ix] , PressTrans_ij[:,ix,0,k]  /Pope_normFactor_ij[:,ix,0,k]    ,'g',linewidth=1)
                plt.plot( y/delta99[ix] , TurbTrans_ij[:,ix,0,k]   /Pope_normFactor_ij[:,ix,0,k]    ,'c')
                plt.plot( y/delta99[ix] , ViscDiff_ij[:,ix,0,k]    /Pope_normFactor_ij[:,ix,0,k]    ,'grey')
                plt.plot( y/delta99[ix] , Conv_ij[:,ix,0,k]        /Pope_normFactor_ij[:,ix,0,k]    ,'orange',linewidth=1)
                plt.plot( y/delta99[ix] , Res_ij[:,ix,0,k]         /Pope_normFactor_ij[:,ix,0,k]    ,':k')
            # if k<4:
            #     refdata = path_to_refdata+'/'+fnames_refdata_buds[k]
            #     refdata = os.path.expanduser(refdata)
            #     df = pd.read_csv(refdata, sep='\s+', header=None, comment="%") 
        #         df.iloc[:,7] = -df.iloc[:,7]
        #         # if not k==3:
        #         plt.semilogx(df.iloc[:,1],df.iloc[:,2:]  ,':c')
        #         # else:
        #         #     plt.semilogx(df.iloc[:,1],df.iloc[:,2:]  ,':c')
                plt.xlim([ 0 , 1.1 ])
                plt.ylim([-1, 1])
                plt.title(suptitles[k])
                plt.legend(('Production','Dissipation','PRS','Pres. Trans.','Turb. Tran.','Visc. Diff','Conv','Residual'))


        # figures 7.35-38 (b)
        suptitles = [ "u'u'" , "v'v'" , "w'w'" , "u'v'" ]
        plt.figure(1502)
        for ix in ix_plot: # range(1000,Nxyz[1],200):
            for k in range(4):
                plt.subplot(221+k)
                plt.plot( y/delta_nu[ix] , Prod_ij[:,ix,0,k]        /(u_tau[ix]**3/delta_nu[ix])    ,'k')
                plt.plot( y/delta_nu[ix] , Diss_ij[:,ix,0,k]        /(u_tau[ix]**3/delta_nu[ix])    ,'b')
                plt.plot( y/delta_nu[ix] , PRS_ij[:,ix,0,k]         /(u_tau[ix]**3/delta_nu[ix])    ,'r',linewidth=1)
                plt.plot( y/delta_nu[ix] , PressTrans_ij[:,ix,0,k]  /(u_tau[ix]**3/delta_nu[ix])    ,'g',linewidth=1)
                plt.plot( y/delta_nu[ix] , TurbTrans_ij[:,ix,0,k]   /(u_tau[ix]**3/delta_nu[ix])    ,'c')
                plt.plot( y/delta_nu[ix] , ViscDiff_ij[:,ix,0,k]    /(u_tau[ix]**3/delta_nu[ix])    ,'grey')
                plt.plot( y/delta_nu[ix] , Conv_ij[:,ix,0,k]        /(u_tau[ix]**3/delta_nu[ix])    ,'orange',linewidth=1)
                plt.plot( y/delta_nu[ix] , Res_ij[:,ix,0,k]         /(u_tau[ix]**3/delta_nu[ix])    ,':k')
        #     if k<4:
        #         refdata = path_to_refdata+'/'+fnames_refdata_buds[k]
        #         refdata = os.path.expanduser(refdata)
        #         df = pd.read_csv(refdata, sep='\s+', header=None, comment="%") 
        #         df.iloc[:,7] = -df.iloc[:,7]
        #         # if not k==3:
        #         plt.semilogx(df.iloc[:,1],df.iloc[:,2:]  ,':c')
        #         # else:
        #         #     plt.semilogx(df.iloc[:,1],df.iloc[:,2:]  ,':c')
                plt.xlim([ 0 , 50 ])
                # plt.xlim((0, 5))
                plt.title(suptitles[k])
                plt.legend(('Production','Dissipation','PRS','Pres. Trans.','Turb. Tran.','Visc. Diff','Conv','Residual'))

    #%% Plot 2D budgets as pcolor
    if IF_plot_pcolor:
        suptitles = [ "u'u'" , "v'v'" , "w'w'" , "u'v'" ]
        for k in range(4):

            f = plt.figure(2001+k)
            f.suptitle(suptitles[k])

            plt.subplot(241)
            plt.pcolor( XYZ[:,1000:-100,0,0], XYZ[:,1000:-100,0,1] , Prod_ij[:,1000:-100,0,k]  /(u_tau[1000:-100]**3/delta_nu[1000:-100]) , cmap='seismic' ,shading='nearest' )
            plt.plot( x[1000:-100] , delta99[1000:-100] , ':k' , lw=2 )
            plt.ylim([0,np.max(delta99)])
            plt.clim([-.01,.01])
            # plt.xlabel(r'$x/\delta_{99,in}$')
            plt.ylabel(r'$y/\delta_{99,in}$')
            plt.title('Production')
            plt.colorbar()

            # plt.figure(2001+k)
            plt.subplot(242)
            plt.pcolor( XYZ[:,1000:-100,0,0], XYZ[:,1000:-100,0,1] , Diss_ij[:,1000:-100,0,k]  /(u_tau[1000:-100]**3/delta_nu[1000:-100]) , cmap='seismic' ,shading='nearest' )
            plt.plot( x[1000:-100] , delta99[1000:-100] , ':k' , lw=2 )
            plt.ylim([0,np.max(delta99)])
            plt.clim([-.01,.01])
            # plt.xlabel(r'$x/\delta_{99,in}$')
            # plt.ylabel(r'$y/\delta_{99,in}$')
            plt.title('Dissipation')
            plt.colorbar()

            # plt.figure(2001+k)
            plt.subplot(243)
            plt.pcolor( XYZ[:,1000:-100,0,0], XYZ[:,1000:-100,0,1] , ViscDiff_ij[:,1000:-100,0,k]  /(u_tau[1000:-100]**3/delta_nu[1000:-100]) , cmap='seismic' ,shading='nearest' )
            plt.plot( x[1000:-100] , delta99[1000:-100] , ':k' , lw=2 )
            plt.ylim([0,np.max(delta99)])
            plt.clim([-.01,.01])
            # plt.xlabel(r'$x/\delta_{99,in}$')
            # plt.ylabel(r'$y/\delta_{99,in}$')
            plt.title('Viscous diff.')
            plt.colorbar()

            # plt.figure(2001+k)
            plt.subplot(244)
            plt.pcolor( XYZ[:,1000:-100,0,0], XYZ[:,1000:-100,0,1] , PRS_ij[:,1000:-100,0,k]  /(u_tau[1000:-100]**3/delta_nu[1000:-100]) , cmap='seismic' ,shading='nearest' )
            plt.plot( x[1000:-100] , delta99[1000:-100] , ':k' , lw=2 )
            plt.ylim([0,np.max(delta99)])
            plt.clim([-.01,.01])
            # plt.xlabel(r'$x/\delta_{99,in}$')
            # plt.ylabel(r'$y/\delta_{99,in}$')
            plt.title('Pressure strain')
            plt.colorbar()

            # plt.figure(2001+k)
            plt.subplot(245)
            plt.pcolor( XYZ[:,1000:-100,0,0], XYZ[:,1000:-100,0,1] , PressTrans_ij[:,1000:-100,0,k]  /(u_tau[1000:-100]**3/delta_nu[1000:-100]) , cmap='seismic' ,shading='nearest' )
            plt.plot( x[1000:-100] , delta99[1000:-100] , ':k' , lw=2 )
            plt.ylim([0,np.max(delta99)])
            plt.clim([-.01,.01])
            plt.xlabel(r'$x/\delta_{99,in}$')
            plt.ylabel(r'$y/\delta_{99,in}$')
            plt.title('Pressure trans')
            plt.colorbar()

            # plt.figure(2001+k)
            plt.subplot(246)
            plt.pcolor( XYZ[:,1000:-100,0,0], XYZ[:,1000:-100,0,1] , TurbTrans_ij[:,1000:-100,0,k]  /(u_tau[1000:-100]**3/delta_nu[1000:-100]) , cmap='seismic' ,shading='nearest' )
            plt.plot( x[1000:-100] , delta99[1000:-100] , ':k' , lw=2 )
            plt.ylim([0,np.max(delta99)])
            plt.clim([-.01,.01])
            plt.xlabel(r'$x/\delta_{99,in}$')
            # plt.ylabel(r'$y/\delta_{99,in}$')
            plt.title('Turb. trans.')
            plt.colorbar()

            # plt.figure(2001+k)
            plt.subplot(247)
            plt.pcolor( XYZ[:,1000:-100,0,0], XYZ[:,1000:-100,0,1] , Conv_ij[:,1000:-100,0,k]  /(u_tau[1000:-100]**3/delta_nu[1000:-100]) , cmap='seismic' ,shading='nearest' )
            plt.plot( x[1000:-100] , delta99[1000:-100] , ':k' , lw=2 )
            plt.ylim([0,np.max(delta99)])
            plt.clim([-.01,.01])
            plt.xlabel(r'$x/\delta_{99,in}$')
            # plt.ylabel(r'$y/\delta_{99,in}$')
            plt.title('Convection')
            plt.colorbar()

            # plt.figure(2001+k)
            plt.subplot(248)
            plt.pcolor( XYZ[:,1000:-100,0,0], XYZ[:,1000:-100,0,1] , Res_ij[:,1000:-100,0,k]  /(u_tau[1000:-100]**3/delta_nu[1000:-100]) , cmap='seismic' ,shading='nearest' )
            plt.plot( x[1000:-100] , delta99[1000:-100] , ':k' , lw=2 )
            plt.ylim([0,np.max(delta99)])
            plt.clim([-.01,.01])
            plt.xlabel(r'$x/\delta_{99,in}$')
            # plt.ylabel(r'$y/\delta_{99,in}$')
            plt.title('Residual')
            plt.colorbar()


# %%
    plt.show()