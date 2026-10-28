########################## postprocessing script ############################

# Script to postprocess neko simulation statistics with PySEMTools



from pysemtools.postprocessing.statistics.RS_budgets import compute_and_write_additional_pstat_fields
from pysemtools.postprocessing.statistics.RS_budgets import interpolate_all_stat_and_pstat_fields_onto_points
from pysemtools.postprocessing.statistics.RS_budgets import convert_2Dstats_to_3D

import subprocess as sp
import numpy as np
import scipy as sc
import csv
import os
import h5py
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import pyvista as pv
import pysemtools.interpolation.pointclouds as pcs
import pysemtools.interpolation.utils as interp_utils
from mpi4py import MPI  # equivalent to the use of MPI_init() in C
import json
from TBL_pp_function import *
from TBL_pp_plots import *

from pysemtools.postprocessing.file_indexing import index_files_from_folder
from pysemtools.postprocessing.statistics.time_averaging import average_field_files

from pysemtools.postprocessing.statistics.RS_budgets_interpolatedPoints_notMPI import read_interpolated_stat_hdf5_fields

from pysemtools.postprocessing.statistics.RS_budgets_interpolatedPoints_notMPI import calculate_budgets_in_Cartesian

from pysemtools.postprocessing.statistics.tensor_manipulations import *

from pysemtools.io.ppymech.neksuite import pynekwrite, pynekread
from pysemtools.datatypes.msh import Mesh
from pysemtools.datatypes.field import FieldRegistry


comm = MPI.COMM_WORLD

top_dir = "/lustre/iwst/iwst128h/R100_deg90/reference/hsmg/restart13_poly7"
#top_dir = '/home/woody/iwst/iwst128h/TBL/DirOutflow_325_20_19_planeTBL'
#top_dir = '/Users/marleneziegler/Desktop/5. Semester/Masterarbeit/TBL/650_24_24_curvedTBL_DirTW'
neko_data = top_dir
#neko_data = os.path.join(top_dir + '/neko_output')

stats_dir        = os.path.join(top_dir + '/' + 'postprocess')
stats2D_filename = "batches_fluid_stats00.f00000"
stats3D_filename = "batches_fluid_stats_3D0.f00000"
query_points_fname = "points_2D.hdf5"
sem_mesh_fname = "./batches_fluid_stats_3D0.f00000"
fname_mesh       = stats3D_filename
plot_dir = stats_dir + '/plots'
openfoam_file_dir = stats_dir + '/openfoam_files'


os.chdir(stats_dir) 

fluid_stats_file_head = sp.run("head -c100 /lustre/iwst/iwst128h/R100_deg90/reference/hsmg/restart13_poly7/fluid_stats00.f00000", shell = True, capture_output=True, text = True)
fluid_stats_file_head = fluid_stats_file_head.stdout.split()
stats_start_time = float(fluid_stats_file_head[7])

fname_mesh       = stats3D_filename
fname_mean       = stats3D_filename
fname_stat       = fname_mean
which_code       = "neko"
nek5000_stat_type = ""
if_do_dssum_on_derivatives = False   

if_do_dssum_before_interp        = False    # whether to do dssum before interpolation
if_create_boundingBox_for_interp = False    # whether to create a bounding box around the point cloud
                                            # useful for cases where there is a cluster of points somewhere
if_pass_points_to_rank0_only     = True     # pass all points to rank0 only. this is to avoid passing duplicate points
                                            # alternative is for each rank to generate its own points only, but that is too complex

path_to_files   = stats_dir
Reynolds_number     = 1 / 0.0007

If_convert_to_single = True     # converts the statiscs/budegts data into single precision
If_average           = True     # calculate average in space

#stats_files = [ "fluid_stats1.fld", "fluid_stats2.fld","fluid_stats3.fld", "fluid_stats4.fld"]
stats_files = ["fluid_stats0.fld"]
fname_averaged  = 'averaged_and_renamed_interpolated_fields.hdf5'   # name of the hdf5 written after rehsaping into structured formation and averaging in space
fname_budget    = "pstat3d_format.hdf5"     # name of the hdf5 file containing calculated budgets
other_budget_file = "pstat3d_channel_format.hdf5"  # second HDF5 (channel) for comparison

########################### activate code parts #################################################
# turn on/off the required code parts

do_indexing_averaging= False 
create_pstats = False
create_points_with_stretching= True 
do_interpolation = True
do_averaging = True
calculate_budgets = True
create_budget_plots = True
save_plots = False
save_txt_profiles = False   # write Schlatter-style .txt profiles (vel + budgets) at the plot stations
save_beta_dpds = False      # write beta(s)/dP_e_ds(s) (Clauser pressure-gradient parameter) for the whole case
If_plot_pcolor = False
plot_channel_data = False
extract_fields = False

# ---------------------------------------------------------------------------------
# How u_tau (and hence delta_nu, y+, U+, c_f, Re_tau) is obtained.
#   'fit'      : wall gradient from a no-slip-enforced fit of U_s(n) over the resolved
#                viscous sublayer (y+ < 2). Use this for curved walls.
#   'gradient' : ORIGINAL behaviour -- stored dUidXj at the first point off the wall.
#                Corrupted in curved-wall regions (u_tau ~25% low in the R100_deg30
#                bend, U+ profiles sitting up to 50% above U+ = y+).
# Flip this string and re-run create_budget_plots to go back; nothing else changes.
# On FLAT-plate stations the stored gradient is exact and 'fit' differs from it by
# up to ~0.5% in u_tau (~1% in c_f) -- small, but not zero, so those numbers move too.
# ---------------------------------------------------------------------------------
utau_method = 'fit'

# function to connect files from different simulations runs
def build_consolidated_stats_index(comm, data_path, stats_file_types, stat_start_time,
                                    output_folder="./", out_name="fluid_stats_index.json"):

    from pysemtools.postprocessing.file_indexing import index_files_from_folder
    index_files_from_folder(comm, folder_path=data_path, run_start_time=0,
    stat_start_time=stat_start_time, output_folder=output_folder,
    file_type=list(stats_file_types), include_file_contents=True,
    include_time_interval=True, overwrite=True)
    out_path = os.path.join(output_folder, out_name)
    if comm.Get_rank() == 0:
        entries = []
        for ftype in stats_file_types:
            idx_name = os.path.join(output_folder,
            os.path.basename(ftype).split(".")[0] + "_index.json")
            if not os.path.exists(idx_name): # skip series not present on disk
                continue
            with open(idx_name) as f:
                idx = json.load(f)
            series = sorted((v for v in idx.values()
                             if isinstance(v, dict) and "time" in v),
                            key=lambda e: e["time"])
            for j, e in enumerate(series):   # chain inside this series only
                prev = series[j-1]["time"] if j > 0 else e["time"]  # j==0 -> interval 0
                e["time_previous_output"] = prev
                e["time_interval"] = e["time"] - prev
            entries += series

        entries.sort(key=lambda e: e["time"]) # global time order, for readability
        consolidated = {"statistics_start_time": stat_start_time}
        for i, e in enumerate(entries):
            consolidated[i] = e
        total = sum(e["time_interval"] for e in entries)
        print(f"consolidated stats index: {len(entries)} files, "
              f"total averaging time = {total:.3f}")
        with open(out_path, "w") as f:
            json.dump(consolidated, f, indent=4)
    comm.Barrier()
    return out_path


# index files and average field files 
if do_indexing_averaging: 
    consolidated_index = build_consolidated_stats_index(comm, neko_data, stats_files, stats_start_time,
                                    output_folder="./", out_name="fluid_stats_index.json")
    average_field_files(comm, field_index_name=consolidated_index, output_folder=stats_dir, output_name="batches_fluid_stats0", output_batch_t_len=50000, mesh_index="0", dtype=np.single, rel_tol=0.05, output_word_size=4, write_mesh=True)


if extract_fields:
    fname_msh = "batches_fluid_stats00.f00000"
    output_file = "velocity_profile.csv"
    # intialize the mesh and some fields
    msh = Mesh(comm, create_connectivity=False)
    fld = FieldRegistry(comm)

    # read mesh
    file_path = stats_dir + "/" + fname_msh
    pynekread(file_path, comm, msh=msh, fld = fld, data_dtype=np.single)
    for key in fld.fields.keys():
        print(f'{key} has {len(fld.fields[key])} fields')
    
    u = fld.fields['vel'][0]
    v = fld.fields['vel'][1]
    #w = fld.fields['vel'][2]
    p = fld.fields['pres'][0]
    t = fld.fields['temp'][0]
    
    print('u shape', u.shape)
    

if create_pstats:
    convert_2Dstats_to_3D(
        stats_dir+"/"+stats2D_filename,
        stats_dir+"/"+stats3D_filename,
        datatype='single') 

    compute_and_write_additional_pstat_fields(
        stats_dir,
        fname_mesh,
        fname_mean,
        fname_stat,
        if_write_mesh=True,
        which_code=which_code,
        nek5000_stat_type=nek5000_stat_type,
        if_do_dssum_on_derivatives=if_do_dssum_on_derivatives)


# create interpolation points
# check that geometry of created points fits current mesh geometry
# needs to be adapted in TBL_pp_functions.py
if create_points_with_stretching:
    if comm.Get_rank() == 0:
        xyz , Nstruct_from_function = user_defined_interpolating_points_highCurv() #use function that fits the curretn geometry
        
    else:
        xyz = 0


if do_interpolation:
    interpolated_fields_output_fname =  os.path.join(stats_dir, "interpolated_fields.hdf5")
    interpolate_all_stat_and_pstat_fields_onto_points(
        stats_dir,
        fname_mesh,
        fname_mean,
        fname_stat,
        xyz,
        which_code=which_code,
        nek5000_stat_type=nek5000_stat_type,
        if_do_dssum_before_interp=if_do_dssum_before_interp,
        if_create_boundingBox_for_interp=if_create_boundingBox_for_interp,
        if_pass_points_to_rank0_only=if_pass_points_to_rank0_only,
        interpolation_output_fname=interpolated_fields_output_fname,
        find_points_tol = 1e-7
    ) 
    

if do_averaging:
    if comm.Get_rank() == 0:
        Nstruct     = Nstruct_from_function # structured formation of the interpolation mesh

    def av_func(x):
        x = sc.ndimage.gaussian_filter1d(x, 7,   axis=1)  # x-Richtung
        x = sc.ndimage.gaussian_filter1d(x, 1.5, axis=0)  # y-Richtung (wandnormal)
        return x

    if comm.Get_rank() == 0:
        read_interpolated_stat_hdf5_fields( 
                path_to_files , 
                Reynolds_number , 
                If_average , 
                If_convert_to_single , 
                Nstruct , 
                av_func , 
                output_fname = fname_averaged
                ) 


if calculate_budgets:
    if comm.Get_rank() == 0:
        calculate_budgets_in_Cartesian(
                path_to_files   = path_to_files , 
                input_filename  = fname_averaged ,
                output_filename = fname_budget )
 

       
if create_budget_plots:

        #plot_dir = stats_dir + '/' + 'comparison_plots'
    input_filename = stats_dir+'/'+fname_budget
    other_input_filename = stats_dir+'/'+other_budget_file
   # budget_plots_old(input_filename, other_input_filename, save_plots, plot_channel_data)
    
    IF_plot_pcolor                  = False
    IF_plot_separate_press_buds     = False
    IF_plot_budgets_in_Pope_style   = False
    IF_plot_budgets_on_top          = True

    plot_ReThetas = [ 2000 ]
    kkk = 0
    # for path_to_files in path_to_files_all:
    #     # input_filename = path_to_files+'/'+fname_budget
    #     input_filename = os.path.expanduser(path_to_files)
        #kkk += 1
    if comm.Get_rank() == 0:    
        with h5py.File(input_filename, "r") as input_file:
            XYZ         = np.array(input_file['XYZ'])
            v1, v2, v3 = define_vectors(XYZ)
            Q           = define_rotation_tensor_from_vectors(v1,v2,v3) 
            #hier vlt reihenfolge ändern wg YXZ??
            Qtrans      = np.transpose(Q,axes=(0,1,2 , 4,3 ))
            norm_Q = np.zeros([62,1920,1,1])
            norm_Q = np.sqrt(np.sum(Q**2,axis=-1,keepdims=False))

            Nxyz        = XYZ.shape
            P       = np.array(input_file['P'])
            UVW     = np.array(input_file['UVW'])
            Rij     = np.array(input_file['Rij'])
            dUidXj  = np.array(input_file['dUidXj'])

            
            Prod_ij         = np.array(input_file['Prod_ij']) 
            Diss_ij         = np.array(input_file['Diss_ij'])
            PRS_ij          = np.array(input_file['PRS_ij']) 
            Conv_ij         = np.array(input_file['Conv_ij']) 
            PressTrans_ij   = np.array(input_file['PressTrans_ij']) 
            TurbTrans_ij    = np.array(input_file['TurbTrans_ij']) 
            VelPressGrad_ij = np.array(input_file['VelPressGrad_ij']) 
            ViscDiff_ij     = np.array(input_file['ViscDiff_ij'])

            from TBL_turbtrans_rebuild import rebuild_turbtrans
            TurbTrans_ij = rebuild_turbtrans(stats_dir + '/' + fname_averaged, XYZ)
            
            # ============================================================================
            # OPTIONAL FIX for the near-wall "wiggles" in the derivative-heavy budget
            # terms (viscous diffusion first and foremost, then turbulent transport).
            #
            # WHY THE WIGGLES ARE THERE:
            #   The tensors above are read straight from Neko's *pre-differentiated*
            #   statistics (dUiUjdx_struct, d2UiUjdx2_struct, ...). Those were computed on
            #   the spectral-element mesh and are only C0-continuous across element faces.
            #   Interpolating them onto this structured wall-normal line leaves a
            #   deterministic kink at every element face the line crosses -> the coherent,
            #   fixed-y+ ripples (identical from station to station, worst on viscous
            #   diffusion because it needs 2nd derivatives; ~2x larger inside the curve).
            #
            # THE IDEA OF THE FIX:
            #   Re-build the derivative terms from the *primitive* interpolated field
            #   Rij = <u_i' u_j'> (stored un-differentiated) by differentiating it HERE,
            #   on the smooth structured grid, with a local-polynomial (Savitzky-Golay-
            #   type) operator that also low-pass filters -> the element-face kinks are
            #   removed instead of amplified. No re-run of Neko needed.
            #
            # SCOPE / CAVEATS:
            #   * Only terms that are pure functions of the primitives can be rebuilt.
            #     Viscous diffusion = nu * Laplacian(Rij) is the cleanest AND the wiggliest,
            #     so it is done fully below. pysemtools builds it component-wise
            #     (ViscDiff_ij[...,c] = (1/Re) * d^2 Rij[...,c]/dx_k^2), so differentiating
            #     each Rij component independently reproduces it with no index bookkeeping.
            #   * Dissipation and the pressure terms (PRS / pressure transport) need
            #     velocity-gradient and pressure-velocity correlations that are NOT
            #     available as primitives here -> leave Neko's versions untouched.
            #   * Near the wall the wall-normal 2nd derivative d^2/dn^2 dominates the
            #     Laplacian (profile varies fast in n, slowly in s) and that is exactly
            #     where the wiggle lives, so only the wall-normal part is rebuilt. The
            #     streamwise part d^2/ds^2 is small there; add it along axis 1 (with the
            #     grid metric) if you need it further from the wall.
            #   * Sanity-check the rebuilt term against the original before trusting it
            #     (same shape/peak, just de-rippled).
            # ============================================================================
            recompute_viscdiff_from_primitives = False   # True -> use the rebuilt term below

            def smooth_normal_derivative(field, n, deriv=1, half_window=6, polyorder=3):
                """d^deriv/dn^deriv of `field` along the wall-normal axis (axis 0),
                evaluated on the NON-uniform physical coordinate `n` (the geometric-
                stretching offsets, identical at every streamwise station).

                For every node a degree-`polyorder` polynomial is least-squares fitted to
                the 2*half_window+1 neighbouring nodes in the local coordinate (n - n_j)
                and differentiated analytically. This respects the stretched grid (no
                uniform-spacing assumption) and smooths at the same time, so it does not
                amplify the interpolation kinks the way a raw finite difference would.
                Any trailing shape works; components are differentiated independently.
                """
                from math import factorial
                field = np.asarray(field, dtype=np.float64)
                ny    = field.shape[0]
                flat  = field.reshape(ny, -1)                 # (ny, Ncols)
                out   = np.zeros_like(flat)
                for j in range(ny):
                    lo, hi   = max(0, j - half_window), min(ny, j + half_window + 1)
                    loc      = n[lo:hi] - n[j]                # local coord centred at node j
                    A        = np.vander(loc, polyorder + 1, increasing=True)  # (win, p+1)
                    coef, *_ = np.linalg.lstsq(A, flat[lo:hi], rcond=None)     # (p+1, Ncols)
                    out[j]   = factorial(deriv) * coef[deriv]  # d-th derivative at loc = 0
                return out.reshape(field.shape)

            if recompute_viscdiff_from_primitives:
                nu       = 1.0 / np.float64(input_file['Rer_here'][()])
                Rij_prim = np.array(input_file['Rij'], dtype=np.float64)   # <u_i'u_j'>, 6 comp
                # wall-normal distance along the interpolation line (same offsets at every
                # station -> take it from station 0); this is the physical n for d/dn.
                n_coord  = np.linalg.norm(XYZ[:, 0, 0, :] - XYZ[0, 0, 0, :], axis=-1)
                # viscous diffusion, component-wise: nu * d^2 Rij/dn^2  (wall-normal part)
                ViscDiff_ij = nu * smooth_normal_derivative(Rij_prim, n_coord,
                                                            deriv=2, half_window=6, polyorder=3)
                print('ViscDiff_ij rebuilt from primitive Rij (smoothed wall-normal Laplacian).')

                # --- the other derivative terms follow the SAME recipe (sketch) -----------
                # Turbulent transport  TurbTrans_ij = -d/dx_k <u_i'u_j'u_k'> :
                #   read the primitive triple products (UiUjUk, 10 indep. comps -- confirm
                #   the component layout against pysemtools first), project the wall-normal
                #   part <u_i'u_j'u_n'> and differentiate once, e.g.
                #   TurbTrans_ij[...,c] = -smooth_normal_derivative(<u_i'u_j'u_n'>, n_coord, deriv=1)
                # Convection            Conv_ij = -U_k dR_ij/dx_k :
                #   dRdn = smooth_normal_derivative(Rij_prim, n_coord, deriv=1)
                #   U_n  = np.sum(UVW * v2, axis=-1, keepdims=True)   # wall-normal velocity (v2)
                #   Conv_ij(normal part) = -U_n * dRdn   (convection is mostly streamwise,
                #                          so add the U_s dR/ds part along axis 1 too)
                # -------------------------------------------------------------------------

            # rotate all budget terms
            UVW_rot     = rotate_vector(UVW,Qtrans,Q)
            UVW_rot_norm = np.zeros_like(UVW_rot)
            UVW_rot_norm =  UVW_rot/ np.sqrt(np.sum(UVW_rot**2,axis=-1,keepdims=True))
            
            Rij_rot = rotate_6c2D_tensor(Rij,Qtrans,Q)
            Prod_ij_rot = rotate_6c2D_tensor(Prod_ij,Qtrans,Q)
            Diss_ij_rot = rotate_6c2D_tensor(Diss_ij,Qtrans,Q)
            PRS_ij_rot = rotate_6c2D_tensor(PRS_ij,Qtrans,Q)
            Conv_ij_rot = rotate_6c2D_tensor(Conv_ij,Qtrans,Q)
            PressTrans_ij_rot = rotate_6c2D_tensor(PressTrans_ij,Qtrans,Q)
            TurbTrans_ij_rot = rotate_6c2D_tensor(TurbTrans_ij,Qtrans,Q)
            VelPressGrad_ij_rot = rotate_6c2D_tensor(VelPressGrad_ij,Qtrans,Q)
            ViscDiff_ij_rot = rotate_6c2D_tensor(ViscDiff_ij,Qtrans,Q)

            VelPressGrad_ij = PressTrans_ij_rot + PRS_ij_rot
            Res_ij          = Prod_ij_rot + Diss_ij_rot + PRS_ij_rot + PressTrans_ij_rot + TurbTrans_ij_rot + ViscDiff_ij_rot + Conv_ij_rot
            Pope_normFactor_ij = np.sqrt( Prod_ij_rot**2 + Diss_ij_rot**2 + PRS_ij_rot**2 + PressTrans_ij_rot**2 + TurbTrans_ij_rot**2 + ViscDiff_ij_rot**2 + Conv_ij_rot**2 )

            dUidXj  = np.reshape( dUidXj , (Nxyz[0],Nxyz[1],Nxyz[2],3,3) , order="F" )
            dUidXj_rot  = rotate_2D_tensor(dUidXj , Qtrans,Q)

            Rer = np.float64(input_file['Rer_here'])
            
            P_meany = np.mean(P, axis = 0)
            print('Rer=',Rer)
        # print('P mean shape:', P_meany.shape)

        #print('UVW shape:', UVW.shape )
        
        #%% compute BL quantities
        x = XYZ[0,:,0,0]
        y = XYZ[:,0,0,1]
        rho = 1


        fluid_nu, tauW_over_rho, u_tau, delta_nu = calculate_friction_units(
            Rer, dUidXj_rot, y=y, UVW_rot=UVW_rot, method=utau_method)

    
        delta99,delta99_approx ,  dStar,dStar_trunc , theta,theta_trunc , Uinf, H12 = calculate_TBL_quantities( x , y , UVW_rot[:,:,0,:] )
        #print('delta 99', delta99)
        #%% compute different Reynolds numbers
        Re_tau , Re_delta99, Re_dStar, Re_theta = calculate_reynolds_numbers(delta99, delta_nu, Uinf, fluid_nu, dStar, theta)


        # streamwise arc length along the wall
        y_wall = XYZ[0,:,0,1]
        ds = np.sqrt(np.diff(x)**2 + np.diff(y_wall)**2)
        s = np.zeros_like(x)
        s[1:] = np.cumsum(ds)

        #%% choose line-plot locations from target Re_theta values
        #plot_ReThetas = np.array([1000, 1200, 1600])
        plot_ReThetas = np.array([300, 700, 1000, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000])   # <-- edit: Re_theta values to plot at
        ix_plot   = np.array([np.argmin(np.abs(Re_theta - rt)) for rt in plot_ReThetas],
                             dtype=np.int32)   # grid index nearest each target
        plot_spos = s[ix_plot]                 # arc length at the selected points
        plot_Re_theta = Re_theta[ix_plot]      # Re_theta actually realised at those points
        print('ix_plot          ', ix_plot)
        print('s at plots        ', plot_spos)
        print('Re_theta at plots ', plot_Re_theta)

        
        # save budget terms in txt-file for certain Re_theta
        if save_txt_profiles:
            export_dir = os.path.join(stats_dir, 'exported_profiles')
            for ix in ix_plot:
                station_dir = os.path.join(export_dir, f'Re_{Re_theta[ix]:.0f}')
                write_reference_style_profiles(
                    station_dir, [ix], y,
                    delta99, delta_nu, u_tau, Uinf,
                    Re_theta, Re_tau, Re_dStar, H12, s,
                    UVW_rot, Rij_rot,
                    Prod_ij_rot, Conv_ij_rot, Diss_ij_rot,
                    PRS_ij_rot, PressTrans_ij_rot, TurbTrans_ij_rot,
                    ViscDiff_ij_rot, Res_ij)

        # save Clauser parameter beta in txt-file
        if save_beta_dpds:
            export_dir = os.path.join(stats_dir, 'exported_profiles')
            write_beta_dpds(export_dir, s, dStar, tauW_over_rho, P[0, :, 0, 0], Re_theta,
                             Uinf=Uinf, Re_tau=Re_tau, Re_dStar=Re_dStar, Re_delta99=Re_delta99,
                             delta99=delta99, theta=theta, H12=H12, u_tau=u_tau, delta_nu=delta_nu)

        print('delta_nu.shape', delta_nu.shape)
        # plots
        os.chdir(plot_dir)
        names = ["uu", "vv", "ww", "uv"]
        #plot_xx(y, delta_nu, u_tau, Rij_rot)
       # plot_pressure(x, P_meany)
        plot_y_height = 10.0   # wall-normal height (physical units) to sample u along x
        #plot_U_over_x(UVW, x, XYZ, plot_y_height)          # original (un-rotated) velocity
        #plot_bernoulli(x, y, UVW, P, rho)
        #plot_Urot_over_x(UVW_rot, x, XYZ, plot_y_height)
       # plot_u(y, delta99, UVW_rot, Uinf,ix_plot, plot_spos, plot_Re_theta)
        plot_tbl_thickness(x,delta99, delta99_approx, dStar, dStar_trunc, theta, theta_trunc)
        plot_reynolds_numbers(x, Re_dStar, Re_theta)

        s_arc = [325.0, 325.0 + 610.0*np.deg2rad(30.0)]
       # plot_pressure_over_s(s, P[0,:,0,0], s_marks=s_arc)
       # plot_dpds_over_s(s, P[0,:,0,0], s_marks=s_arc)
        #plot_clauser(s, dStar, tauW_over_rho, P[0,:,0,0])   # lower wall, see write_beta_dpds note above


        #plot_delta_pressure_over_s(s, P[0,:,0,0], s_anchor=s_arc[0], s_marks=s_arc)
        plot_Ue_over_s(s, Uinf, s_anchor=s_arc[0], s_marks=s_arc)
        #plot_accel_param_over_s(s, Uinf, fluid_nu, smooth_len=150.0, s_marks=s_arc)
        plot_near_wall_u(y, delta_nu, UVW_rot, u_tau, ix_plot, plot_spos, plot_Re_theta)
        plot_fluctuations(y, delta_nu, u_tau, Rij_rot, ix_plot, plot_spos, plot_Re_theta)
        #plot_budgets(plot_spos, names, ix_plot, y, delta_nu, u_tau, Prod_ij_rot, Conv_ij_rot, Diss_ij_rot,PRS_ij_rot, PressTrans_ij_rot , TurbTrans_ij_rot, ViscDiff_ij_rot, Res_ij, plot_Re_theta)
        plot_k_budgets(Nxyz, names, ix_plot, y, delta_nu, u_tau, Prod_ij_rot, Conv_ij_rot, Diss_ij_rot,PRS_ij_rot, PressTrans_ij_rot , TurbTrans_ij_rot, ViscDiff_ij_rot, Res_ij, plot_Re_theta)