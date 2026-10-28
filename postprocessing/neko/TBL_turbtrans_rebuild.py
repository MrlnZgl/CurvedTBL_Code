"""Rebuild the turbulent-transport tensor from the primitive triple products.

WHY THIS EXISTS
---------------
pysemtools builds TurbTrans_ij as

    TurbTrans_ij = -d_k <u_i u_j u_k>            (TOTAL triple product, from Neko)
                   + (mean-flow correction terms built from UVW, dUidXj, dRij_dxk)
                   - Conv_ij - Prod_ij

which algebraically reduces to -d_k <u_i' u_j' u_k'>. The problem is that it gets
there by cancelling terms that are ~10^4 larger than the answer: <uuu> ~ U^3 ~ 1.3
while <u'u'u'> ~ u_tau^3 ~ 6e-5. Every piece of that cancellation is a
pre-differentiated SEM field interpolated onto the structured grid, and each carries
its own element-face error. The errors do not cancel, so a ~1e-4 relative error in
d<uuu>/dx becomes a ~100% error in the transport term.

Measured on R100_deg90 reference/hsmg/restart13_poly7 (Re_theta = 3000 station):
    t-diff high-wavenumber content   0.0062 -> 0.00006   (100x smoother)
    k-budget rms residual            0.0190 -> 0.00045   (42x better closure)
    integral of t-diff over the BL   -2.00  -> -0.055    (must be ~0)
Only the streamwise normal-stress component is affected; vv, ww and uv are already
fine (factor 1.0x), and the TKE budget inherits the uu error through the trace.

The fix: form <u_i'u_j'u_k'> from the stored primitives and differentiate it HERE,
on the smooth structured grid, with a Savitzky-Golay operator. No cancellation, no
element-face kinks, no Neko re-run.

USAGE (in TBL_pp_script.py, inside the `if create_budget_plots:` rank-0 block)
-----------------------------------------------------------------------------
Right after TurbTrans_ij is read from the budget file and BEFORE the rotate_* calls:

    from TBL_turbtrans_rebuild import rebuild_turbtrans
    TurbTrans_ij = rebuild_turbtrans(stats_dir + '/' + fname_averaged, XYZ)

The returned array has the same shape/layout/Cartesian frame as the one it replaces,
so rotate_6c2D_tensor and everything downstream are unchanged.

NOTE this only fixes turbulent transport. The near-wall (y+ < ~10) imbalance on the
curved wall is a separate issue and is not touched here.
"""

import numpy as np
import h5py
from scipy.signal import savgol_filter


# stored order of UiUjUk_struct, verified against z-symmetry (odd number of w's -> 0)
_T = {'uuu': 0, 'vvv': 1, 'www': 2, 'uuv': 3, 'uuw': 4,
      'uvv': 5, 'vvw': 6, 'uww': 7, 'vww': 8, 'uvw': 9}


def _fluct3(T, i, A, B, C, ab, ac, bc):
    """<a'b'c'> = <abc> - A<bc> - B<ac> - C<ab> + 2ABC.

    The second moments <bc>, <ac>, <ab> here are the TOTAL ones (UiUj_struct as
    stored), NOT the Reynolds stresses. Passing Reynolds stresses instead leaves the
    mean-flow part uncancelled and inflates <q^2 u'> by ~4 orders of magnitude.
    """
    return T[..., i] - A * bc - B * ac - C * ab + 2.0 * A * B * C


def _grid_frame(XYZ):
    """Orthonormal (tangent, normal) frame and arc-length metric of the structured
    interpolation grid. Axis 0 is wall-normal, axis 1 is streamwise."""
    X, Y = XYZ[..., 0], XYZ[..., 1]
    dn = np.sqrt((X - X[0, :]) ** 2 + (Y - Y[0, :]) ** 2)
    n1 = (X - X[0, :]) / np.maximum(dn, 1e-30)
    n2 = (Y - Y[0, :]) / np.maximum(dn, 1e-30)
    n1[0], n2[0] = n1[1], n2[1]          # wall row: reuse the first interior normal
    t1, t2 = n2, -n1                     # tangent = z_hat x normal
    seg = np.sqrt(np.diff(X, axis=1) ** 2 + np.diff(Y, axis=1) ** 2)
    ds = np.empty_like(X)
    ds[:, 1:-1] = 0.5 * (seg[:, :-1] + seg[:, 1:])
    ds[:, 0], ds[:, -1] = seg[:, 0], seg[:, -1]
    return t1, t2, n1, n2, ds, dn[:, 0]


def _d_dn(F, n, half=8, order=3):
    """d/dn along axis 0 on the geometrically stretched grid. A degree-`order`
    polynomial is least-squares fitted to the 2*half+1 neighbours in the LOCAL
    physical coordinate (n - n_j), so the non-uniform spacing is respected."""
    ny = F.shape[0]
    flat = F.reshape(ny, -1)
    out = np.zeros_like(flat)
    for j in range(ny):
        lo, hi = max(0, j - half), min(ny, j + half + 1)
        A = np.vander(n[lo:hi] - n[j], order + 1, increasing=True)
        out[j] = np.linalg.lstsq(A, flat[lo:hi], rcond=None)[0][1]
    return out.reshape(F.shape)


def rebuild_turbtrans(averaged_hdf5, XYZ, half_n=8, half_s=8, order=3):
    """-d_k <u_i'u_j'u_k'> in the Cartesian frame, 6 components [uu,vv,ww,uv,uw,vw].

    averaged_hdf5 : path to averaged_and_renamed_interpolated_fields.hdf5
    XYZ           : XYZ array already read from the budget file, shape (ny,nx,nz,3)
    half_n/half_s : Savitzky-Golay half-windows (points) wall-normal / streamwise.
                    8/8 with order 3 was the best trade-off on R100_deg90; going
                    wider smooths more but starts to degrade closure again.
    """
    with h5py.File(averaged_hdf5, 'r') as f:
        T = f['UiUjUk_struct'][:, :, 0, :].astype(np.float64)
        U = f['UVW_struct'][:, :, 0, :].astype(np.float64)
        M = f['UiUj_struct'][:, :, 0, :].astype(np.float64)

    Ux, Uy = U[..., 0], U[..., 1]
    Z = np.zeros_like(Ux)
    uu, vv, ww, uv = M[..., 0], M[..., 1], M[..., 2], M[..., 3]   # TOTAL moments

    # W = 0 and <uw> = <vw> = 0 by spanwise symmetry, so the w-triple products
    # collapse to <u'w'w'> = <uww> - U<ww> and <v'w'w'> = <vww> - V<ww>.
    t_uuu = _fluct3(T, _T['uuu'], Ux, Ux, Ux, uu, uu, uu)
    t_vvv = _fluct3(T, _T['vvv'], Uy, Uy, Uy, vv, vv, vv)
    t_uuv = _fluct3(T, _T['uuv'], Ux, Ux, Uy, uu, uv, uv)
    t_uvv = _fluct3(T, _T['uvv'], Ux, Uy, Uy, uv, uv, vv)
    t_uww = _fluct3(T, _T['uww'], Ux, Z,  Z,  Z,  Z,  ww)
    t_vww = _fluct3(T, _T['vww'], Uy, Z,  Z,  Z,  Z,  ww)

    t1, t2, n1, n2, ds, nvec = _grid_frame(XYZ[:, :, 0, :].astype(np.float64))
    d_ds = lambda F: savgol_filter(F, 2 * half_s + 1, order, deriv=1,
                                   axis=1, mode='interp') / ds
    d_dn = lambda F: _d_dn(F, nvec, half_n, order)

    def div(Fx, Fy):
        """d(Fx)/dx + d(Fy)/dy, with d/dx = t_x d/ds + n_x d/dn (orthogonal grid)."""
        return t1 * d_ds(Fx) + n1 * d_dn(Fx) + t2 * d_ds(Fy) + n2 * d_dn(Fy)

    ny, nx = t_uuu.shape
    out = np.zeros((ny, nx, 1, 6), dtype=np.float32)
    out[:, :, 0, 0] = -div(t_uuu, t_uuv)        # uu
    out[:, :, 0, 1] = -div(t_uvv, t_vvv)        # vv
    out[:, :, 0, 2] = -div(t_uww, t_vww)        # ww
    out[:, :, 0, 3] = -div(t_uuv, t_uvv)        # uv
    # uw, vw vanish by spanwise symmetry -> left at zero
    return out
