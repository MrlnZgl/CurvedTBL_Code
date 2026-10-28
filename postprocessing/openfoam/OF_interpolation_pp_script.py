
import re
import numpy as np
from fluidfoam import readmesh
import csv
from mpi4py import MPI #equivalent to the use of MPI_init() in C
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree
from pysemtools.datatypes.msh import Mesh
from pysemtools.datatypes.coef import Coef
from pysemtools.datatypes.field import Field, FieldRegistry
from pysemtools.io.ppymech.neksuite import preadnek, pynekread
from pysemtools.datatypes.utils import write_fld_subdomain_from_list
from pysemtools.datatypes.coef import Coef
from pysemtools.datatypes.utils import extrude_2d_sem_mesh
import pysemtools.interpolation.utils as interp_utils
from os.path import join
from pysemtools.interpolation.probes import Probes
from OF_interpolation_functions import *
from TBL_pp_function import *
#from TBL_pp_plots import *
from pysemtools.postprocessing.statistics.RS_budgets_interpolatedPoints_notMPI import read_interpolated_stat_hdf5_fields
from pysemtools.postprocessing.statistics.RS_budgets import interpolate_all_stat_and_pstat_fields_onto_points
from pysemtools.postprocessing.statistics.RS_budgets import convert_2Dstats_to_3D
from pysemtools.postprocessing.statistics.RS_budgets import compute_and_write_additional_pstat_fields
from pysemtools.postprocessing.statistics.RS_budgets_interpolatedPoints_notMPI import calculate_budgets_in_Cartesian
from pysemtools.postprocessing.statistics.tensor_manipulations import *
np.set_printoptions(threshold=np.inf)


top_dir = "/lustre/iwst/iwst128h/TBL/R100_deg30/referenceTW/fast_pol7_restart_stats_650_24_24"
#top_dir = '/home/woody/iwst/iwst128h/TBL/DirOutflow_325_20_19_planeTBL'
neko_data = top_dir
#neko_data = os.path.join(top_dir + '/neko_output')

stats_dir        = os.path.join(top_dir + '/' + 'openfoam_postprocess')
stats2D_filename = "batches_fluid_stats00.f00000"
stats3D_filename = "batches_fluid_stats_3D0.f00000"
sem_mesh_fname = "./batches_fluid_stats_3D0.f00000"
fname_mesh       = stats3D_filename
fname_mean       = stats3D_filename
fname_stat       = fname_mean
plot_dir = stats_dir + '/plots'
openfoam_file_dir = top_dir + '/openfoam_files'
openfoam_plots = stats_dir + '/plots'
fname_budget    = "pstat3d_format.hdf5"
os.chdir(stats_dir)

which_code       = "neko"
nek5000_stat_type = ""
if_do_dssum_on_derivatives = False   

if_do_dssum_before_interp        = False    # whether to do dssum before interpolation
if_create_boundingBox_for_interp = False    # whether to create a bounding box around the point cloud
                                            # useful for cases where there is a cluster of points somewhere
if_pass_points_to_rank0_only     = True     # pass all points to rank0 only. this is to avoid passing duplicate points
 

CASE    = "/lustre/iwst/iwst128h/TBL/R100_deg30/optimizedTW/pressure_tests/OFcase"

#DATA_PATH = "/lustre/iwst/iwst128h/TBL/650_20_12_curvedTBL_DirTW_outflow_long/postprocess"
case_name = "."

bottomwall_name = "bottomwall"
inflow_name = "inflow"
do_interpolation = True
read_budgets_file = True

#xyz, Nstruct = read_openfoam_interpol_point(CASE, bottomwall_name)
#x_wall, y_wall, z_wall = readmesh(CASE, boundary=bottomwall_name)

x_in, y_in, z_in = readmesh(CASE, boundary=inflow_name)
y_indeces = y_in.argsort()
y_in = np.sort(y_in)
y_in_unsorted = y_in[y_indeces]
print(x_in, y_in, z_in)
print('yindeces', y_indeces)
print('y unsorted', y_in_unsorted)

# Cut off OF mesh points above DNS domain height (DNS tops out at y=24.5)
DNS_y_max = 24.5
y_cutoff_mask = y_in <= DNS_y_max
ny_full = y_cutoff_mask.size          # total number of OF inflow faces (before cutoff)
y_in = y_in[y_cutoff_mask]
y_indeces = y_indeces[y_cutoff_mask]
n_upper = ny_full - len(y_in)         # number of faces cut off (y > DNS_y_max)
print(f"Cutoff at y={DNS_y_max}: keeping {y_cutoff_mask.sum()} of {ny_full} points ({n_upper} upper faces will be padded)")

## vlt lieber nicht sortieren??
#y_in = np.sort(y_in)
x_inf = 0
nx = 1
ny = len(y_in)
nz = 1

x_Retheta_offset = 275
x_inf += x_Retheta_offset
Nstruct = [ny, nx, nz]
X, Y = np.meshgrid(x_inf, y_in)
Z = np.zeros_like(X)
xyz = interp_utils.transform_from_array_to_list(ny, nx, nz, [X, Y, Z])
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if comm.Get_rank() == 0:
    xyz = xyz
else:
    xyz = None

if do_interpolation:
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

    interpolated_fields_output_fname =  os.path.join(stats_dir, "interpolated_fields.hdf5")
    #if comm.Get_rank() == 0:
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
    path_to_files   = stats_dir
    Reynolds_number     = 1 / 0.0007

    If_convert_to_single = True     # converts the statiscs/budegts data into single precision
    If_average           = True     # calculate average in space


    fname_averaged  = 'averaged_and_renamed_interpolated_fields.hdf5'
    def av_func(x):
        # return x
        return sc.ndimage.gaussian_filter( x , 5 , axes=1 )

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

    
        
        calculate_budgets_in_Cartesian(
                        path_to_files   = path_to_files , 
                        input_filename  = fname_averaged ,
                        output_filename = fname_budget )
if comm.Get_rank() == 0: 
    if read_budgets_file:
        if comm.Get_rank() == 0: 
            input_filename = stats_dir+'/'+fname_budget
            with h5py.File(input_filename, "r") as input_file:
                XYZ         = np.array(input_file['XYZ'])
 
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
                
             
                dUidXj  = np.reshape( dUidXj , (Nxyz[0],Nxyz[1],Nxyz[2],3,3) , order="F" )
            
                Rer = np.float64(input_file['Rer_here'])
                
                P_meany = np.mean(P, axis = 0)
                print('Rer=',Rer)
            
            
            print('duDX shape', dUidXj.shape)
            fluid_nu, tauW_over_rho, u_tau, delta_nu = calculate_friction_units(Rer, dUidXj)
            print('delta_nu', delta_nu)
            print('u_tau', u_tau)
            print('Rij shape', Rij.shape)
            #%% compute BL quantities
            x = XYZ[0,:,0,0]
            y = XYZ[:,0,0,1]
            rho = 1
            #print('x',XYZ[0,1500:,0,0] )
            #  print('y', y)

            delta99,delta99_approx ,  dStar,dStar_trunc , theta,theta_trunc , Uinf, H12 = calculate_TBL_quantities( x , y , UVW[:,:,0,:] )
            print('theta', theta)
            #print('delta 99', delta99)
            #%% compute different Reynolds numbers
            Re_tau , Re_delta99, Re_dStar, Re_theta = calculate_reynolds_numbers(delta99, delta_nu, Uinf, fluid_nu, dStar, theta)
            y_plus = y/delta_nu
            print('y_plus', y_plus)

            # extract values from statistics
            os.chdir(openfoam_file_dir)
            desired_Retheta = 1000
            index = find_index(desired_Retheta, Re_theta)
            print('delta99 at index', delta99[index])
            print('x at index Re theta 1000', x[index])
            extract_tke_at_xposition(index, u_tau,Rij, y_indeces)
            extract_vel_at_xposition(index, UVW, y_indeces)
            extract_omega_at_xposition(index,Rij, Diss_ij, u_tau, delta_nu, y_indeces)
 