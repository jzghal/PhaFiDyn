#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=====================================================================
   Phase field Unified Formulation - CZM model
       >> Fenics implementaion code :
           o Dynamic simulation
           o Implemented without History Variable
=====================================================================

@code contributors:            
    Jihed ZGHAL (jzghal@parisnanterre.fr)
    Abdelmalek BARKI (barki.malek97@gmail.com)
    Nesrine AISSA (nesrine.aissa2112@gmail.com)
    
@Organization:      LEME-EA4416, Paris Nanterre University
@Fenics Version:    2019.2.0.dev0

"""

# ---- Import modules and python envirement ----
from __future__ import print_function
import time, datetime
StratPgm = time.process_time()

import math, sympy, shutil
import os, sys, logging
import numpy as np


from dolfin import *
from fenics import *
import ufl

# ------------------ 
# Parameters
# ------------------
set_log_level(LogLevel.ERROR)  # log level
# set some dolfin specific parameters
info(parameters,True)
parameters.parse()    # Read parameters from the command line
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["quadrature_degree"] = 4

snes_Rtol         = 1e-7 # relative tolerance for SNES solver (phase field eq.)
snes_Atol         = 1e-9 # absolute tolerance for SNES solver (phase field eq.)
snes_maxiter      = 1000   # max. iteration for SNEs solver (phase field eq.)

# ---- Mesh informations
# Configure information about the mesh
path_file = ''
base_file = 'PhaFiDyn_Branch'
convert = True

# ---- Material informations
# Configure material properties
# Configure material properties
E    = Constant(32e9)         # [Pa], Young modulus
nu   = Constant(0.2)            # Poisson ratio
Gc   = Constant(3)          # [Pa.m], Toughness
sigc = (27/256/2.5e-4*Gc*E)**(0.5)           # Contrainte critique (ou limite elastique) [Pa]
l_ch = E*Gc/sigc**2
l = 5e-4 
l = float(l)
rho = Constant(2450)
v_r = 2125 # [m/s] Rayleigh wave speeds

"""
hypothses >> plane strain : plane_strain
          >> Plane stress : plane_stress
"""
hypothese = "plane_strain"

"""
Material behavior >> Sym
                  >> DevDecompHyb
                  >> DevDecomp
                  >> SpecDecompHyb
                  >> SpecDecomp 
"""
Behav     = "DevDecomp"

# ---- Phase field parameters
# Configure phase field parameters
"""
CZM Type:
    - Linear_softening
    - Exponential_softening
    - Bilinear_softening
    - Hyperbolic_softening
    - Cornelissen_softening
"""
Model    = 'CZM'
CZM_Type = 'Linear_softening'

# ---- Results informations
Nout = 1000             # Save output on paraview each Nout Files


# ---- Output parameters

# Time of excution of the program: 
#   - Generate the new file name with the current date, hour, and minute
StratPgm = time.process_time()
current_datetime = datetime.datetime.now().strftime("%Y-%m-%d-%Hh%Mmin%Ssec")

Res_folder = './Results/'+base_file+'_True'+Model+'_'+current_datetime+'/'    # Dossier resultats
if not os.path.exists(Res_folder):
    os.makedirs(Res_folder)



# ---- Fonctions and classes ----

# Matematical operator
def Max(a, b):
	return (a+b+abs(a-b))/Constant(2)

def Min(a, b):
	return (a+b-abs(a-b))/Constant(2)

def PositivePart(f):
	r = 0.5*(f + abs(f))
	return r

def NegativePart(f):
	r = 0.5*(f - abs(f))
	return r



# Mesh files
def MeshImport (path_file, base_file, convert):
    """
    Inputs:
      - path_file     >> path of mesh file
      - base_file     >> Name of the mesh file
      - convert       >> True  : convert the gmsh file to fenics xml file
                      >> False : No conversion is made
    Outputs:
      - mesh          >> mesh of the studied structure
      - Regions_zone  >> Regions zone in the mesh of the structure (used for
                         multimaterial)
      - Boundary_zone >> Boundary zone in the mesh of the structure : define
                         the boudary condition
      - ndim          >> Dimension of the problem
    """
    
    if (convert):
        print("- Convert mesh ...")
        cmd = 'gmsh -3 '+path_file+base_file+'.geo -format msh2'
        os.system(cmd)
        
        cmd = 'dolfin-convert '+path_file+base_file+'.msh '+path_file+base_file+'.xml'
        os.system(cmd)
        
        print("- Convert mesh ...    Done !!")
        
    print("- Import mesh ... ", end='')
    mesh_file     = path_file + base_file + '.xml'
    mesh          = Mesh(mesh_file)

    Regions_file  = path_file + base_file + '_physical_region.xml'
    Regions_zone  = MeshFunction("size_t", mesh, Regions_file)

    Boundary_file = path_file + base_file + '_facet_region.xml'
    Boundary_zone = MeshFunction('size_t', mesh, Boundary_file)

    ndim = mesh.topology().dim()        # get number of space dimensions
    dx = Measure("dx", subdomain_data=Regions_zone)          # Compute integration variable over the domain
    n  = FacetNormal(mesh)
    
    print("Done !!")
    
    return (mesh, Regions_zone, Boundary_zone, ndim, dx, n)


# ---- ----     Post process     ---- ----
def ResultsExport(base_file, current_datetime):
    """
    Res_folder : Folder where results will be saved
    Res_file   : Paraview file where results will be saved
    """
    
    #  --- Paraview Files
    Res_file = Res_folder+base_file+'.xdmf'     # Results file
    
    # ---- Paraview File
    Paraview_File = XDMFFile(Res_file)
    Paraview_File.parameters["flush_output"] = True
    Paraview_File.parameters["functions_share_mesh"] = True
    
    return ( Paraview_File )

def postprocessing(Incr, Res_folder, base_file, Data):
    elastic_energy_value = assemble(Elastic_energy)
    kinetic_energy_value = assemble(Kinetic_energy)
    surface_energy_value = assemble(Dissipated_energy)
    external_energy_value = ext_energy

    total_energy_value = surface_energy_value + elastic_energy_value + kinetic_energy_value - external_energy_value  
    
    
    temp_data = np.zeros(16)
    
    temp_data[0] = Incr                  # >> Increment
    temp_data[1] = t                     # >> Time
    temp_data[2] = elastic_energy_value  # >> Elastic energy
    temp_data[3] = kinetic_energy_value  # >> Kinetic energy
    temp_data[4] = surface_energy_value  # >> Fracture energy
    temp_data[5] = external_energy_value # >> External energy
    temp_data[6] = total_energy_value    # >> Total Energy
    
    temp_data[7] = phi.vector().min()    # Minimum damage
    temp_data[8] = phi.vector().max()    # Maximum damage

    # Find the indices where phi >= 0.95 for the upper and lower branches
    mask_crack_top = (phi.compute_vertex_values() >= 0.95) & (mesh.coordinates()[:, 1] >= 0.)
    mask_crack_bot = (phi.compute_vertex_values() >= 0.95) & (mesh.coordinates()[:, 1] <= 0.)

    if np.any(mask_crack_top):
        # Extract the corresponding coordinates for the top
        coords_top = mesh.coordinates()[mask_crack_top]
        # Find the maximum x-coordinate
        index_top = np.argmax(coords_top[:, 0])
        # Coordinates of the upper crack tip
        crack_tip_top = coords_top[index_top]
    else:
        crack_tip_top = (0, 0)

    if np.any(mask_crack_bot):
        # Extract the corresponding coordinates for the bottom
        coords_bot = mesh.coordinates()[mask_crack_bot]
        # Find the maximum x-coordinate
        index_bot = np.argmax(coords_bot[:, 0])
        # Coordinates of the lower crack tip
        crack_tip_bot = coords_bot[index_bot]
    else:
        crack_tip_bot = (0, 0)

    temp_data[9] = crack_tip_top[0]  # x_dmax 
    temp_data[10] = crack_tip_top[1]  # y_dmax 

    temp_data[11] = crack_tip_bot[0]  # x_dmax 
    temp_data[12] = crack_tip_bot[1]  # y_dmax

        
    temp_data[13] = ERROR  # Error Linf    
    temp_data[14] = timestep  # Duration of a timestep
    temp_data[15] = float(dt)  # Delta t
    
    # Define the column header for the output data
    lst = 'Increment \t Time \t ElasticEnergy \t KineticEnergy \t FractureEnergy \t ExternalEnergy \t TotalEnergy \t dmin \t dmax \t x_dmax_top \t y_dmax_top \t x_dmax_bot \t y_dmax_bot \t error_Linf \t Timestep \t dt'
    
    # Save the results to a data file
    np.savetxt(Res_folder + base_file + '.data', Data, delimiter='\t', header=lst)
    Data.append(temp_data)
    del temp_data
    
def ParaviewResults(Incr, Res_file):
    """
    Export simulation results to Paraview format.

    Parameters:
    - Incr: Incremental step or time frame for the results.
    - Res_file: The Paraview file where results will be saved.

    This function exports several fields, including Displacement, Damage, Stress, and Strain,
    to a Paraview file for visualization and analysis.

    Args:
    - Incr (float): Incremental step or time frame.
    - Res_file (XDMFFile): Paraview file to write results.

    Returns:
    None
    """
    # Create a Function to represent Displacement
    disp = Function(Vdisp, name='Displacement')
    disp.assign(u)

    # Write Displacement data to the Paraview file
    Paraview_File.write(disp, Incr)

    # Create a Function to represent Velocity
    Vel = Function(Vdisp, name='Velocity')
    Vel.assign(v)

    # Write Velocity data to the Paraview file
    Paraview_File.write(Vel, Incr)

    # Create a Function to represent Acceleration
    Acc = Function(Vdisp, name='Acceleration')
    Acc.assign(a)

    # Write Acceleration data to the Paraview file
    Paraview_File.write(Acc, Incr)

    # Create a Function to represent Damage
    damage = Function(Vdam, name='Damage')
    damage.assign(phi)

    # Write Damage data to the Paraview file
    Paraview_File.write(damage, Incr)

    # Create a TensorFunctionSpace for Stress
    T = TensorFunctionSpace(mesh, 'CG', 1)

    # Project the Stress field
    cont = project(sigma(u, phi), T)
    Contrainte = Function(T, name='Stress')
    Contrainte.assign(cont)

    # Write Stress data to the Paraview file
    Paraview_File.write(Contrainte, Incr)

    # Project the Strain field
    defor = project(epsilon(u), T)
    deform = Function(T, name='Strain')
    deform.assign(defor)

    # Write Strain data to the Paraview file
    Paraview_File.write(deform, Incr)
      


def Backup(path_file, base_file, Res_folder, current_datetime):
    """
    Create a backup of the current script file and copy mesh files to the results folder.

    Parameters:
    - path_file: Path to the current script file.
    - base_file: Base file name for saving results.
    - Res_folder: Folder where results will be saved.
    - current_datetime: Current date and time for creating a unique backup file.

    This function creates a backup of the current script file by copying it with a unique
    timestamp and moves the backup to the specified results folder. It also copies mesh files
    with the given base file name to the results folder.

    Args:
    - path_file (str): Path to the current script file.
    - base_file (str): Base file name for saving results.
    - Res_folder (str): Folder where results will be saved.
    - current_datetime (str): Current date and time for timestamping the backup file.

    Returns:
    None
    """
    # Get the current script file path
    current_file_path = os.path.realpath(__file__)

    # Extract the file name and extension
    file_name, file_extension = os.path.splitext(current_file_path)

    # Create a new file name with a timestamp
    new_file_name = f"{file_name}_{current_datetime}{file_extension}"

    # Copy the current script file to the backup file
    shutil.copy(file_name + file_extension, new_file_name)

    # Move the backup file to the results folder
    shutil.move(new_file_name, Res_folder)

    # Copy mesh files (geo and msh) to the results folder
    FileName = path_file + base_file + ".geo"
    shutil.copy(FileName, Res_folder)

    FileName = path_file + base_file + ".msh"
    shutil.copy(FileName, Res_folder)




    
   

# --- Material
def MaterialDef(E, nu, Gc, hypothese):
    """
    Define material properties.

    Parameters:
    - E (float): Young's modulus.
    - nu (float): Poisson's ratio.
    - Gc (float): Critical energy release rate (fracture toughness).
    - hypothese (str): Material behavior hypothesis, e.g., "plane_stress".

    This function calculates and returns material properties including the shear modulus (mu),
    Lamé's first parameter (lmbda), and the bulk modulus (K) based on input parameters.
    If the hypothesis is "plane_stress," it modifies Lamé's first parameter accordingly.

    Args:
    - E (float): Young's modulus.
    - nu (float): Poisson's ratio.
    - Gc (float): Critical energy release rate (fracture toughness).
    - hypothese (str): Material behavior hypothesis.

    Returns:
    - mu (float): Shear modulus.
    - lmbda (float): Lamé's first parameter.
    - K (float): Bulk modulus.
    """

    # Calculate shear modulus and Lamé's first parameter
    mu = E / (2 * (1 + nu))
    lmbda = (E * nu) / ((1 + nu) * (1 - 2 * nu))

    # Modify Lamé's first parameter for plane stress hypothesis
    if hypothese == "plane_stress":
        lmbda = 2 * mu * lmbda / (lmbda + 2 * mu)

    # Calculate bulk modulus
    K = lmbda + 2 * mu / ndim

    return (mu, lmbda, K)

def epsilon(u):
    """Calculate the strain tensor as a function of displacement."""
    return sym(grad(u))


def sigma_0(u):
    """Calculate the stress tensor of the undamaged material as a function of displacement."""
    return (2.0 * mu * (epsilon(u)) + lmbda * tr(epsilon(u)) * Identity(ndim))



"""
u        :  Displacement
Behavior : material behavior hypothesis
    - Sym       : Symmetric behavior (Bourdin et al, https://doi.org/10.1016/S0022-5096(99)00028-9))
    - DevDecompHyb : Deviatoric decomposition of \Psi_0 (Amor et al, https://doi.org/10.1016/j.jmps.2009.04.011)  and \sigma = g(d) \sigma_0 (Marreddy Ambati et al, http://dx.doi.org/10.1007/s00466-014-1109-y)
    - DevDecomp :  Deviatoric decomposition of \Psi_0 (Amor et al, https://doi.org/10.1016/j.jmps.2009.04.011) 
"""

# Checking if Behav is equal to 'Sym'
if Behav == 'Sym':
    def sigma(u, d):
        """
        Calculate the stress tensor of the damaged material as a function of displacement and damage.

        Parameters:
        - u: Displacement field.
        - d: Damage field.

        Returns:
        - Stress tensor for the damaged material.
        """

        return (g(d) * sigma_0(u))
    
    # Definition of the Psi0 function
    def Psi0(u, Behav):
        """
        Calculate the strain energy density function for symmetric material behavior.

        Parameters:
        - u: Displacement field.
        - Behav: Material behavior type.

        Returns:
        - Psi: Strain energy density function.
        """
        # Calculate Psi using the given parameters
        Psi = 0.5 * lmbda * (tr(epsilon(u)))**2 + mu * inner(epsilon(u), epsilon(u))
        return Psi

    
if Behav == 'DevDecompHyb':
    def sigma(u, d):
        """
        Calculate the stress tensor of the damaged material as a function of displacement and damage.

        Parameters:
        - u: Displacement field.
        - d: Damage field.

        Returns:
        - Stress tensor for the damaged material.
        """

        return (g(d) * sigma_0(u))
    
    def Psi0(u, Behav):
        """
        Calculate the strain energy density function for deviatoric decomposition material behavior.

        Parameters:
        - u: Displacement field.
        - Behav: Material behavior type.

        Returns:
        - Tuple (PsiP, PsiN): Strain energy density functions for positive and negative parts.
        """
        PsiP = 0.5 * K * PositivePart(tr(epsilon(u)))**2 + mu * inner(dev(epsilon(u)), dev(epsilon(u)))
        PsiN = 0.5 * K * NegativePart(tr(epsilon(u)))**2
        return (PsiP, PsiN)


if Behav == 'DevDecomp':
    # Definition of the Psi0 function
    def Psi0(u, Behav):
        """
        Calculate the strain energy density function for deviatoric decomposition material behavior.

        Parameters:
        - u: Displacement field.
        - Behav: Material behavior type.

        Returns:
        - Tuple (PsiP, PsiN): Strain energy density functions for positive and negative parts.
        """
        PsiP = 0.5 * K * PositivePart(tr(epsilon(u)))**2 + mu * inner(dev(epsilon(u)), dev(epsilon(u)))
        PsiN = 0.5 * K * NegativePart(tr(epsilon(u)))**2
        return (PsiP, PsiN)
    
    def sigma(u, d):
        """
        Calculate the stress tensor for a damaged material as a function of displacement and damage.

        Parameters:
            - u: Displacement field.
            - d: Damage field.

        Returns:
        - The stress tensor for the damaged material.
        """
        # Calculate the strain tensor epsilon(u) from the displacement field u
        eps = epsilon(u)

        # Calculate the positive part of the trace of epsilon(u)
        plus_tr_eps = PositivePart(tr(eps))

        # Calculate the negative part of the trace of epsilon(u)
        moins_tr_eps = NegativePart(tr(eps))

        # Calculate psi_plus_prime, a stress-related quantity, based on material properties
        psi_plus_prime = K * plus_tr_eps * Identity(ndim) + 2 * mu * dev(eps)

        # Calculate psi_moins_prime, another stress-related quantity, based on material properties
        psi_moins_prime = K * moins_tr_eps * Identity(ndim)

        # Calculate the stress tensor s by combining psi_plus_prime and psi_moins_prime
        s = g(d) * psi_plus_prime + psi_moins_prime

        return s


if Behav == 'SpecDecompHyb':
    def sigma(u, d):
        """
        Calculate the stress tensor of the damaged material as a function of displacement and damage.

        Parameters:
        - u: Displacement field.
        - d: Damage field.

        Returns:
        - Stress tensor for the damaged material.
        """

        return (g(d) * sigma_0(u))
    
    tol_v = 1e-24
    # Positive part of the decomposed strain
    def eps_positive(u):    
        A = sym(grad(u))
        a = A[0, 0]
        b = A[0, 1]
        c = A[1, 0]
        d = A[1, 1]
        eig_1 = ((tr(A) + sqrt(tr(A)**2 - 4 * det(A) + tol_v)) / 2)
        eig_2 = ((tr(A) - sqrt(tr(A)**2 - 4 * det(A) + tol_v)) / 2)
        phi_1 = (eig_1 - b - d) / (a + c - eig_1)
        phi_2 = (eig_2 - b - d) / (a + c - eig_2)

        eig_v_1 = [phi_1 / sqrt(phi_1**2 + 1), 1 / sqrt(phi_1**2 + 1)]
        eig_v_2 = [phi_2 / sqrt(phi_2**2 + 1), 1 / sqrt(phi_2**2 + 1)]

        # Positive Strain Tensor
        sn_P = 0.5 * (eig_1 + abs(eig_1)) * np.outer(eig_v_1, eig_v_1) + \
               0.5 * (eig_2 + abs(eig_2)) * np.outer(eig_v_2, eig_v_2)
        sn_1 = as_matrix(sn_P.tolist())
        return sn_1

    # Negative part of the decomposed strain
    def eps_negative(u):    
        A = sym(grad(u))
        a = A[0, 0]
        b = A[0, 1]
        c = A[1, 0]
        d = A[1, 1]
        eig_1 = ((tr(A) + sqrt(tr(A)**2 - 4 * det(A) + tol_v)) / 2)
        eig_2 = ((tr(A) - sqrt(tr(A)**2 - 4 * det(A) + tol_v)) / 2)
        phi_1 = (eig_1 - b - d) / (a + c - eig_1)
        phi_2 = (eig_2 - b - d) / (a + c - eig_2)

        eig_v_1 = [phi_1 / sqrt(phi_1**2 + 1), 1 / sqrt(phi_1**2 + 1)]
        eig_v_2 = [phi_2 / sqrt(phi_2**2 + 1), 1 / sqrt(phi_2**2 + 1)]

        # Negative Strain Tensor
        sn_N = 0.5 * (eig_1 - abs(eig_1)) * np.outer(eig_v_1, eig_v_1) + \
               0.5 * (eig_2 - abs(eig_2)) * np.outer(eig_v_2, eig_v_2)
        sn_1 = as_matrix(sn_N.tolist())
        return sn_1

    # Definition of the Psi0 function
    def Psi0(u, Behav):
        """
        Calculate the strain energy density function for spectral decomposition material behavior.

        Parameters:
        - u: Displacement field.
        - Behav: Material behavior type.

        Returns:
        - Tuple (PsiP, PsiN): Strain energy density functions for positive and negative parts.
        """
        epsp = eps_positive(u)
        epsn = eps_negative(u)
        PsiP = 0.5 * lmbda * PositivePart(tr(epsilon(u)))**2 + mu * inner(epsp, epsp)
        PsiN = 0.5 * lmbda * NegativePart(tr(epsilon(u)))**2 + mu * inner(epsn, epsn)
        return (PsiP, PsiN)
        

if Behav == 'SpecDecomp':
    tol_v = 1e-24
    # Positive part of the decomposed strain
    def eps_positive(u):    
        A = sym(grad(u))
        a = A[0, 0]
        b = A[0, 1]
        c = A[1, 0]
        d = A[1, 1]
        eig_1 = ((tr(A) + sqrt(tr(A)**2 - 4 * det(A) + tol_v)) / 2)
        eig_2 = ((tr(A) - sqrt(tr(A)**2 - 4 * det(A) + tol_v)) / 2)
        phi_1 = (eig_1 - b - d) / (a + c - eig_1)
        phi_2 = (eig_2 - b - d) / (a + c - eig_2)

        eig_v_1 = [phi_1 / sqrt(phi_1**2 + 1), 1 / sqrt(phi_1**2 + 1)]
        eig_v_2 = [phi_2 / sqrt(phi_2**2 + 1), 1 / sqrt(phi_2**2 + 1)]

        # Positive Strain Tensor
        sn_P = 0.5 * (eig_1 + abs(eig_1)) * np.outer(eig_v_1, eig_v_1) + \
               0.5 * (eig_2 + abs(eig_2)) * np.outer(eig_v_2, eig_v_2)
        sn_1 = as_matrix(sn_P.tolist())
        return sn_1

    # Negative part of the decomposed strain
    def eps_negative(u):    
        A = sym(grad(u))
        a = A[0, 0]
        b = A[0, 1]
        c = A[1, 0]
        d = A[1, 1]
        eig_1 = ((tr(A) + sqrt(tr(A)**2 - 4 * det(A) + tol_v)) / 2)
        eig_2 = ((tr(A) - sqrt(tr(A)**2 - 4 * det(A) + tol_v)) / 2)
        phi_1 = (eig_1 - b - d) / (a + c - eig_1)
        phi_2 = (eig_2 - b - d) / (a + c - eig_2)

        eig_v_1 = [phi_1 / sqrt(phi_1**2 + 1), 1 / sqrt(phi_1**2 + 1)]
        eig_v_2 = [phi_2 / sqrt(phi_2**2 + 1), 1 / sqrt(phi_2**2 + 1)]

        # Negative Strain Tensor
        sn_N = 0.5 * (eig_1 - abs(eig_1)) * np.outer(eig_v_1, eig_v_1) + \
               0.5 * (eig_2 - abs(eig_2)) * np.outer(eig_v_2, eig_v_2)
        sn_1 = as_matrix(sn_N.tolist())
        return sn_1

    # Definition of the Psi0 function
    def Psi0(u, Behav):
        """
        Calculate the strain energy density function for spectral decomposition material behavior.

        Parameters:
        - u: Displacement field.
        - Behav: Material behavior type.

        Returns:
        - Tuple (PsiP, PsiN): Strain energy density functions for positive and negative parts.
        """
        epsp = eps_positive(u)
        epsn = eps_negative(u)
        PsiP = 0.5 * lmbda * PositivePart(tr(epsilon(u)))**2 + mu * inner(epsp, epsp)
        PsiN = 0.5 * lmbda * NegativePart(tr(epsilon(u)))**2 + mu * inner(epsn, epsn)
        return (PsiP, PsiN)
    
    def sigma(u, d):
        """
        Calculate the stress tensor for a damaged material as a function of displacement and damage.

        Parameters:
            - u: Displacement field.
            - d: Damage field.

        Returns:
        - The stress tensor for the damaged material.
        """
        # Calculate the strain tensor epsilon(u) from the displacement field u
        eps = epsilon(u)

        # Calculate the positive and negative part of the strain tensor epsilon(u)
        epsp = eps_positive(u)
        epsn = eps_negative(u)
        
        # Calculate psi_plus_prime, a stress-related quantity, based on material properties
        psi_plus_prime = lmbda * PositivePart(tr(eps)) * Identity(ndim) + 2 * mu * epsp

        # Calculate psi_moins_prime, another stress-related quantity, based on material properties
        psi_moins_prime = lmbda * NegativePart(tr(eps)) * Identity(ndim)  + 2 * mu * epsn

        # Calculate the stress tensor s by combining psi_plus_prime and psi_moins_prime
        s = g(d) * psi_plus_prime + psi_moins_prime

        return s
 
    
    
 # ---- Phase Field Functions
# PF-CZM softening parameters
if bool(CZM_Type):
    a1 = 4 * l_ch / math.pi / l
    if (CZM_Type=='Linear_softening'):
        pp = 2
        k0 = (-1*(sigc**2))/(2*Gc)
        a2 = -1/2
        a3 = 0
    elif (CZM_Type=='Exponential_softening'):
        pp = 5/2
        k0 = (-1*(sigc**2))/Gc
        a2 = 2.**(5/3)-3
        a3 = 0
    elif (CZM_Type=='Bilinear_softening'):
        pp = 5/2
        k0 = (-1*(sigc**2))/Gc
        a2 = 0.03687
        a3 = 20.8343
    elif (CZM_Type=='Hyperbolic_softening'):
        pp = 4
        k0 = (-2*(sigc**2))/Gc  
        a2 = 2.**(7/3)-9/2
        a3 = 0
    elif (CZM_Type=='Cornelissen_softening'):
        pp = 2
        k0 = (-1.3546*(sigc**2))/Gc
        a2 = 1.3868
        a3 = 0.6567

      
def alpha(d, Model):
    """
    Calculate the geometric crack function alpha.

    Parameters:
    - d (float): Phase field variable representing damage.
    - Model (str): Model type, e.g., 'CZM' 

    Returns:
    - alfa (float): Geometric crack function value.
    """
    if (Model == 'CZM'):
        alpha_phi   = 2*d - d**2
    
    return alpha_phi


def alpha_p(d, Model):
    """
    Calculate the derivative of the geometric crack function alpha with respect to phase field.

    Parameters:
    - d (float): Phase field variable representing damage.
    - Model (str): Model type, e.g., 'CZM'

    Returns:
    - alpha_phi_p (float): Derivative of the geometric crack function.
    """
    if (Model == 'CZM'):
        alpha_phi_p   = 2 - 2*d
    
    return alpha_phi_p


def alpha_pp(d, Model):
    """
    Calculate the second derivative of the geometric crack function alpha with respect to phase field.

    Parameters:
    - d (float): Phase field variable representing damage.
    - Model (str): Model type, e.g., 'CZM'.

    Returns:
    - alpha_phi_pp (float): Second derivative of the geometric crack function.
    """
    if (Model == 'CZM'):
        alpha_phi_pp   =  -2
    
    return alpha_phi_pp


def g_pp(d):
    """
    Calculate the second derivative of the degradation function g with respect to phase field.

    Parameters:
    - d (float): Phase field variable representing damage.

    Returns:
    - g_phi_pp (float): Second derivative of the degradation function.
    """
    if (Model == 'CZM'):
        
        Num = (1-d)**pp
        dNum  = -pp*(1 - d)**(pp - 1)
        ddNum = pp*(pp-1)*(1-d)**(pp-2)
        
        
        Q_d   = a1*d + a1*a2*d**2 + a1*a2*a3*d**3
        Den = Num + Q_d
        dDen  = dNum + a1 + 2*a1*a2*d + 3*a1*a2*a3*d**2
        ddDen = ddNum + 2*a1*a2 + 6*a1*a2*a3*d
       
        
        g_phi_pp = ( (ddNum*Den - Num*ddDen) * Den - (dNum*Den - Num*dDen) * 2 * dDen ) / (Den**3)

    return g_phi_pp


def g_p(d):
    """
    Calculate the derivative of the degradation function g with respect to phase field.

    Parameters:
    - d (float): Phase field variable representing damage.

    Returns:
    - g_phi_p (float): Derivative of the degradation function.
    """
    if (Model == 'CZM'):
        
        Num = (1-d)**pp
        dNum  = -pp*(1 - d)**(pp - 1)

        
        Q_d   = a1*d + a1*a2*d**2 + a1*a2*a3*d**3
        Den = Num + Q_d
        dDen  = dNum + a1 + 2*a1*a2*d + 3*a1*a2*a3*d**2

        
        g_phi_p = (dNum*Den - Num*dDen)  / (Den**2)

    return g_phi_p



def g(d):
    """
    Calculate the degradation function.

    Parameters:
    - d (float): Phase field variable representing damage.

    Returns:
    - gd (float): Degradation function value.
    """
    kappa = Constant(1.e-12)  # Residual stiffness
    
    if (Model == 'CZM'):
        FRAC1 = (1-d)**pp
        P_d   = 1 + ( (a2*d) * (1+a3*d) )
        Q_d   = a1*d*P_d
        FRAC2 = FRAC1 + Q_d
        g_phi = FRAC1/FRAC2
        
    return g_phi +  kappa


def PFparameters(Model):
    """
    Calculate phase field parameters.

    Parameters:
    - E (float): Young's modulus.
    - nu (float): Poisson's ratio.
    - Gc (float): Critical energy release rate (fracture toughness).
    - sigc (float): Characteristic stress.
    - Model (str): Model type, e.g., 'CZM'.

    Returns:
    - Tuple (c0, l): Phase field parameters.
    """
    if (Model == 'CZM'):
        c0 = math.pi
    return c0

# ---- Boundary Conditions Functions     
# Function to apply displacement boundary conditions
def BCDisp (Vdisp, bc_u, IDzone, ux, uy):
    """
    Vdisp    : Fenics FEM vector space which describe displacement
    bc_u     : bondary condition on displacement
    IDzone   : ID of loaded edge (physical edge in GMSH)
    ux, uy   : Imposed displacement on the selected edge through respectively x
               and y direction
    """
    # Fixed zone
    if bool(ux):
        Lx  = DirichletBC( Vdisp.sub(0), ux, Boundary_zone, IDzone )
        bc_u.append(Lx)
        
    if bool(uy):
        Ly  = DirichletBC( Vdisp.sub(1), uy, Boundary_zone, IDzone )
        bc_u.append(Ly)
        
    return (bc_u)
    

# Function to apply damage boundary conditions
def BCDam (Vdam, bc_dam, IDdam, dini):
    """
    Vdam     : Fenics FEM scalar space which describe damage
    bc_dam   : bondary condition on damage
    IDdam    : ID damaged edge (physical edge in GMSH)
    dini     : Initial damage
    """

    # Damage
    if bool(IDdam):
        dam_loc = DirichletBC(Vdam, dini, Boundary_zone, IDdam)
        bc_dam.append(dam_loc)
    
    return (bc_dam)



# ------ ------     Main program     ------ ------ #

# Initialize Paraview file for results export
Paraview_File = ResultsExport(base_file, current_datetime)

"""
   >> Mesh and material parameters
"""
# Import mesh and extract relevant information
mesh, Regions_zone, Boundary_zone, ndim, dx, n = MeshImport(path_file, base_file, convert)

# Calculate material parameters
mu, lmbda, K = MaterialDef(E, nu, Gc, hypothese)

# Calculate phase field parameters
c0 = PFparameters(Model)

# Make a backup of the running code
Backup(path_file, base_file, Res_folder, current_datetime)

# ---------------------------------------
"""
   >> Boundary conditions and loading
"""
# Number of increments and load levels
class Traction_top(UserExpression):
    def __init__(self, t, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.t = t
    def eval(self,values,x):
        values[0] = 0
        if  float(self.t) <= 0.0:
            values[1] = 0
        else:
            values[1] = 1e6
    def value_shape(self):
        return (2,)

# Defining load - Traction bottom
class Traction_botttom(UserExpression):
    def __init__(self, t, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.t = t
    def eval(self,values,x):
        values[0] = 0
        if float(self.t) <= 0.0:
            values[1] = 0
        else:
            values[1] = -1e6
    def value_shape(self):
        return (2,)

load_top = Traction_top(0, degree=1)                                              # Initiating the load
load_bottom = Traction_botttom(0, degree=1)

# IDs for boundary conditions
IDTop, IDBot = 100, 200

T = 8e-5           # Final Time
# # Critical time increment (approximated CFL condition)
c_s = (E/rho*(1-nu)/((1+nu)*(1-2*nu)))**0.5 
explicit_safety_factor = 5 # >1
dt0 = mesh.hmin()/c_s/explicit_safety_factor
dt = Constant(min(float(dt0), 1e-8, mesh.hmin()/v_r))
print("dt = ", float(dt))


# Initial damage
IDdam, dini = [], []

# FEM Fencis
# Define function spaces for displacement, damage
Vdisp   = VectorFunctionSpace(mesh, 'CG', 1)
Vdam    = FunctionSpace(mesh, 'CG', 1)

# Define trial and test functions for the mechanical problem (displacement field)
du, u_ = TrialFunction(Vdisp), TestFunction(Vdisp)

# Define trial and test functions for the phase field problem (phi field)
dphi, q = TrialFunction(Vdam), TestFunction(Vdam)

# Fields from previous time step
u, v, a  = Function(Vdisp), Function(Vdisp), Function(Vdisp)

# Fields from new time step  (displacement, velocity, acceleration)
u_new, v_new, a_new  = Function(Vdisp), Function(Vdisp), Function(Vdisp)

# Damage fields from previous and new step 
phi, pold = Function(Vdam), Function(Vdam)

# Define integration variables for local integration
dx = Measure("dx", subdomain_data=Regions_zone)  # Integration variable over the domain
ds = Measure("ds", domain=mesh, subdomain_data=Boundary_zone, subdomain_id=IDTop)  # Integration over boundary
n  = FacetNormal(mesh)  # Facet normal vector

# Displacement Boundary conditions
bc_u = []
bc_v = []

# Damage Boundary conditions
bc_dam = []

# Define energies based on material behavior
if Behav== 'Sym':
    E_elas = (g(phi) * Psi0(u, Behav)) * dx
else:
    PsiP, PsiN     = Psi0(u, Behav)
    E_elas = ((g(phi)*PsiP)+PsiN)*dx
E_cine = 0.5 * rho * inner(v, v) * dx
E_ext = (dot(load_top, u)*ds(IDTop) + dot(load_bottom, u)*ds(IDBot))  
E_d = (Gc / c0) * ((alpha(phi, Model) / l) + (l * dot(grad(phi), grad(phi)))) * dx

# Energy derivatives
Epot_du = derivative(E_elas, u, u_) - derivative(E_ext, u, u_) 
Ecine_dv = derivative(E_cine, v, u_)

# --------------------- Initialization of PF problem -----------------------
# Damage problem
if Behav== 'Sym':
    PsiP = Psi0 (u, Behav)
else:
    PsiP, PsiN = Psi0 (u, Behav )
EDam_0 = ((g_p(phi) * PsiP * q)
        +
        (Gc / c0 / l * (alpha_p(phi, Model) * q + 2 * l * l * dot(grad(q), grad(phi))))) * dx

J_EDam_0 = ((g_pp(phi) * PsiP * q * dphi)
          +
          (Gc / c0 / l * (alpha_pp(phi, Model) * q * dphi + 2 * l * l * dot(grad(q), grad(dphi))))) * dx
for bc in bc_dam:
    bc.apply(phi.vector())

# Constraints for the damage
d_min = interpolate(Constant(0.), Vdam)  # Lower bound
d_max = interpolate(Constant(1.0), Vdam) # Upper bound
for bc in bc_dam:
    bc.apply(d_min.vector())

# Define the linear and nonlinear variational problems
DamProblem  = NonlinearVariationalProblem(EDam_0,  phi, bc_dam, J_EDam_0)
DamProblem.set_bounds(d_min, d_max) # set bounds for the phase field
# Construct solvers for displacement and damage
DamSolver  = NonlinearVariationalSolver(DamProblem)
# Damage Solver Parameters
DamSolver.parameters.update({
    "nonlinear_solver": "snes",
    "snes_solver": {
        "method": "vinewtonssls",
        "line_search": "cp",
        "maximum_iterations": snes_maxiter,
        "relative_tolerance": snes_Rtol,
        "absolute_tolerance": snes_Atol,
        "report": True,
        "error_on_nonconvergence": False
    }
})
info(DamSolver.parameters, True)

# Disp problem
EDisp_0 = ufl.replace(Ecine_dv, {v: du}) + ufl.replace(Epot_du, {u: u})
for bc in bc_u:
    bc.apply(u.vector())
for bc in bc_v:
    bc.apply(v.vector())

DamSolver.solve()
solve(lhs(EDisp_0)==rhs(EDisp_0), a)
pold.assign(phi)
d_min.vector()[:] = phi.vector()

# --------------------------- Start of PF problem -------------------------
# Disp problem
Edisp = ufl.replace(Ecine_dv, {v: a}) + Epot_du

# Mass matrix
V1 = FunctionSpace(mesh, "CG", 1)
z, y = TestFunction(V1), TrialFunction(V1)
M_lumped = action(z*y*dx, Constant(1))
M_lumped = assemble(M_lumped)
M_doubled = np.repeat(M_lumped.get_local(), 2)

F_Form = Form(ufl.replace(-(Epot_du), {u: u_new}))

Elastic_energy = Form(E_elas)
Kinetic_energy = Form(E_cine)
External_energy = Form(E_ext)
Dissipated_energy = Form(E_d)


# Define Linear Variational Problem (Quasi-static Case)
# Define energies based on material behavior
if Behav== 'Sym':
    PsiP = Psi0 (u_new, Behav)
else:
    PsiP, PsiN = Psi0 (u_new, Behav )
   

# Define Linear Variational Problem for the phase field damage
EDam = ((g_p(phi) * PsiP * q)
        +
        (Gc / c0 / l * (alpha_p(phi, Model) * q + 2 * l * l * dot(grad(q), grad(phi))))) * dx

J_EDam = ((g_pp(phi) * PsiP * q * dphi)
          +
          (Gc / c0 / l * (alpha_pp(phi, Model) * q * dphi + 2 * l * l * dot(grad(q), grad(dphi))))) * dx

# Define the linear and nonlinear variational problems
DamProblem  = NonlinearVariationalProblem(EDam,  phi, bc_dam, J_EDam)
DamProblem.set_bounds(d_min, d_max) # set bounds for the phase field

# Construct solvers for displacement and damage
DamSolver  = NonlinearVariationalSolver(DamProblem)

# Damage Solver Parameters
DamSolver.parameters.update({
    "nonlinear_solver": "snes",
    "snes_solver": {
        "method": "vinewtonssls",
        "line_search": "cp",
        "maximum_iterations": snes_maxiter,
        "relative_tolerance": snes_Rtol,
        "absolute_tolerance": snes_Atol,
        "report": True,
        "error_on_nonconvergence": False
    }
})

info(DamSolver.parameters, True)

# Data Storage and Simulation Loop
Data = []  # List to store simulation data

# Initialization
Incr = 0 
t = Constant(0)
ext_energy = 0.

# Loop over the loads
errphi  = Function(Vdam)

# Loop over the loads
while float(t) < T:
    load_top.t  = t                    
    load_bottom.t = t
    
    print('------------------------------------------------------------- ')
    print('------------------------------------------------------------- ')
    print("Incrément: %2.8g, Time: %2.8g" % (Incr, float(t)))
    print('------------------------------------------------------------- ')
    print('------------------------------------------------------------- ')
    
    start = time.process_time()
    # Solve the elastic problem
    print(' ')
    print('[Solving balance equations...]')

    u_new.vector()[:] = u.vector() + dt*v.vector() + 0.5*dt**2*a.vector()  

    # Solve the damage problem
    print('[Solving phase field equation...]') # find d_{n+1} u=u_{n+1}
    DamSolver.solve()
    errphi.vector()[:] = phi.vector() - pold.vector()
    ERROR = norm(errphi.vector(), "linf")

    F = assemble(F_Form)
    a_new.vector().set_local(F.get_local()/M_doubled/float(rho))

    # Update velocity
    v_new.vector()[:] = v.vector() + 0.5*dt*(a_new.vector() + a.vector())

    # Update old fields with new quantities
    u.vector()[:] = u_new.vector()
    v.vector()[:] = v_new.vector()
    a.vector()[:] = a_new.vector()
    pold.assign(phi)

    # updating the d_min to account for the irreversibility
    d_min.vector()[:] = phi.vector()

    # Monitor the results
    print("Error Linf: %2.8g, phi_max: %.8g" % (ERROR, phi.vector().max()))
    
    timestep = time.process_time() - start
    
    ext_energy += float(dt)*assemble(External_energy)

    # Perform post-processing and data storage
    postprocessing(Incr, Res_folder, base_file, Data)
    
    # Save results to Paraview files at specified intervals
    if (Incr % Nout) == 0 or (float(dt)>=float(T-t)):
        ParaviewResults(Incr, Paraview_File)
    
    # Update time and increment
    t.assign(t + dt)
    dt.assign(float(conditional(lt(dt, T-t), dt, T - t)))
    Incr += 1

# Calculate and display the total running time of the simulation
RunningTime = time.process_time() - StratPgm
print("Simulation completed \n \n")
print("  >> Running time [s] :", RunningTime)

# Store the running time in a file
liste = []
liste.append('Running Time [s] = ')
liste.append(str(RunningTime))
with open(Res_folder + base_file + 'RunningTime.dat', 'w') as filout:
    for lst in liste:
        filout.write(f"{lst}")
