# # 22.06.2025 
# This code creates the sample geometry for the Heat conduction problem and creates an XDMF file with the mesh that is called in the code of Heat3D_dogbone.py #

#=================================
#           LIBRARIES            #
#=================================

from mpi4py import MPI  # MPI library is to carry out parallel computations
import gmsh  # interface to the gmsh geometry
import math  # math.calculations, will be used for the fillet of the geometry
from dolfinx import fem, mesh, io  # io is for writing XMDF files(providing input and ouptut)
from dolfinx.io.gmshio import model_to_mesh  # we need this command in order to convert the gmsh model to a Dolfinx mesh
import numpy as np  # Import numpy for numerical operations
import os  # Import os for interacting with the operating system (e.g., creating directories)

# MPI setup
mesh_comm = MPI.COMM_WORLD  # Get the global MPI communicator
model_rank = 0 # the mesh creation is the main process and gathers all the cores
gdim = 3 # dimension of the problem
fdim = gdim - 1 # dimension of the facets where the boundary conditions will be applied

# Only the model_rank (root) process creates the Gmsh geometry
if mesh_comm.rank == model_rank:
    gmsh.model.add("dogbone_mesh") # Add a new Gmsh model with a given name
    gmsh.model.setCurrent("dogbone_mesh")
    occ = gmsh.model.occ           # Get the OpenCASCADE CAD kernel interface -  this is the kernell that is repsonsilbe for the complex geometry handling

    gmsh.initialize()

    # ============================
    #         Parameters
    # ============================
    unit = 0.001 # all the dimensions are in meters
    gauge_length = 8 * unit
    gauge_diameter = 5 * unit
    end_diameter = 13.5 * unit
    fillet_radius = 13.5 * unit
    end_length_straight = 16 * unit

    h_mesh = 1.2 * unit
    h_fine = 0.01* unit   # thats the size of the element, at the region of the refinement- where the laser scanning is happening

    gaude_radius = gauge_diameter / 2.0
    end_radius = end_diameter / 2.0
    delta_half_width = end_radius - gaude_radius
    dx_fillet = math.sqrt(max(0, delta_half_width * (2 * fillet_radius - delta_half_width)))
    assert dx_fillet > 0

    x_gauge_half = gauge_length / 2.0
    x_fillet_end = x_gauge_half + dx_fillet
    x_outer_end = x_fillet_end + end_length_straight

    # ============================
    #          Geometry
    # ============================

    # the following are the ponins that define the outline of the geometry
    p = [
        occ.add_point(x_outer_end, 0, 0),
        occ.add_point(x_outer_end, end_radius, 0),
        occ.add_point(x_fillet_end, end_radius, 0),
        occ.add_point(x_gauge_half, gaude_radius, 0),
        occ.add_point(-x_gauge_half, gaude_radius, 0),
        occ.add_point(-x_fillet_end, end_radius, 0),
        occ.add_point(-x_outer_end, end_radius, 0),
        occ.add_point(-x_outer_end, 0, 0)
    ]

    # c stands for the centers for all the fillets
    c = [
        occ.add_point(x_gauge_half, gaude_radius + fillet_radius, 0),
        occ.add_point(-x_gauge_half, gaude_radius + fillet_radius, 0)
    ]
    # Define line segments and circular arcs that form the 2D profile
    lines = {
        "right_face": occ.add_line(p[0], p[1]),
        "top_straight_right_end": occ.add_line(p[1], p[2]),
        "fillet_tr": occ.add_circle_arc(p[2], c[0], p[3]),
        "top_gauge": occ.add_line(p[3], p[4]),
        "fillet_tl": occ.add_circle_arc(p[4], c[1], p[5]),
        "top_straight_left_end": occ.add_line(p[5], p[6]),
        "left_face": occ.add_line(p[6], p[7]),
        "bottom_closure": occ.add_line(p[7], p[0])
    }
    # # Create a closed loop from the defined line segments (edges of the 2D profile)
    # This loop outlines the cross-sectional shape of the dogbone specimen
    curve_loop = occ.add_curve_loop([lines[k] for k in lines]) # creates a closed loop of curves that define the boundary of the surface
    
    # Create a planar surface bounded by the curve loop
    # This surface represents the 2D profile that will be revolved into a 3D volume
    surface = occ.add_plane_surface([curve_loop]) # it takes the closed loop and creates a surface inside it 
    # Synchronize the OpenCASCADE kernel with the Gmsh model
    # This is required before performing further operations like revolve or meshing
    occ.synchronize() # commits all the geometry operations

    # ============================
    # Revolve
    # ============================
    # Revolve the 2D surface around the x-axis (0,0,0 as point, 1,0,0 as axis) by 2*pi (360 degrees)
    revolved = occ.revolve([(2, surface)], 0, 0, 0, 1, 0, 0, 2 * math.pi)
    occ.synchronize()

    # Get the tag of the newly created 3D volume

    volume_tag = next((e[1] for e in revolved if e[0] == 3), None) # that tells python to find the first revolved entity that is a volume 
    assert volume_tag

    # ============================
    # Box refinement
    # ============================
    # Define the lines you want to refine around: same as in your first code
    # --- Radial refinement ---
    # ===============================
    # 1. Pick the gauge lines for refinement
    refined_lines = [lines["top_gauge"]]

    # ===============================
    # 2. Create a band surface by extruding the gauge line to give it axial thickness
    # We'll extrude the top gauge line symmetrically ± along Z to make a thin surface band.

    band_half_length = 0.3 * unit  # 0.5 mm up, 0.5 mm down → 1 mm band
    extrusions = []

    for l in refined_lines:
        # Extrude up
        up = occ.extrude([(1, l)], 0, 0, band_half_length)
        extrusions.extend(up)
        # Extrude down
        down = occ.extrude([(1, l)], 0, 0, -band_half_length)
        extrusions.extend(down)

    occ.synchronize()

    # Collect new surface tags
    band_surfaces = [e[1] for e in extrusions if e[0] == 2]

    # ===============================
    # 3. Distance field to the band surfaces - standard mesh refinement steps-check Corrando Maurini tutorials on fracture mechanics
    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "SurfacesList", band_surfaces)
    gmsh.model.mesh.field.setNumber(1, "NumPointsPerCurve", 100)

    # ===============================
    # 4. Threshold for smooth gradation
    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin", h_fine)
    gmsh.model.mesh.field.setNumber(2, "SizeMax", h_mesh)
    gmsh.model.mesh.field.setNumber(2, "DistMin", 0.0)
    gmsh.model.mesh.field.setNumber(2, "DistMax", 1.0 * unit)  # Smooth transition over 1 mm radially

    # ===============================
    # 5. Set this as the background mesh
    gmsh.model.mesh.field.setAsBackgroundMesh(2)

    # ===============================   
    # 6. Still enforce global max mesh size as a fallback
    gmsh.option.setNumber("Mesh.MeshSizeMax", h_mesh)

    # ============================
    # Physical groups
    # ============================
    gmsh.model.add_physical_group(3, [volume_tag], tag=1) # mark the volume with the ID=1   
    gmsh.model.set_physical_name(3, 1, "DogboneVolume")

    # Get all outer surfaces (faces) of the 3D volume so we can label them individually
    bnds = gmsh.model.get_boundary([(3, volume_tag)], oriented=False, combined=False)
    tol = 1e-6
    # Lists to store surface tags based on their location
    left_face, right_face, laser_face, top_surf = [], [], [], []

    # Loop through each boundary surface and determine its position using center of mass
    for dim, tag in bnds:
        com = gmsh.model.occ.get_center_of_mass(dim, tag)
        if math.isclose(com[0], -x_outer_end, abs_tol=tol):
            left_face.append(tag)
        elif math.isclose(com[0], x_outer_end, abs_tol=tol):
            right_face.append(tag)
        elif abs(com[0]) < x_fillet_end:
            top_surf.append(tag)
        else:
            laser_face.append(tag)

    # assign physical groups to surfaces- labelling surfaces so they can be clearly identified
    # Label the left-end surface group with tag=2 and name it 'LeftEnd'
    gmsh.model.add_physical_group(2, left_face, tag=2)
    gmsh.model.set_physical_name(2, 2, "LeftEnd")

    # Label the right-end surface with tag=3 and name it 'RightEnd'
    gmsh.model.add_physical_group(2, right_face, tag=3)
    gmsh.model.set_physical_name(2, 3, "RightEnd")

    # Label surfaces affected by laser with tag=4 and name it 'LaserSurface'
    gmsh.model.add_physical_group(2, laser_face, tag=4)
    gmsh.model.set_physical_name(2, 4, "LaserSurface")
    
    # Label the top surfaces (center area) with tag=5 and name it 'TopSurface'
    gmsh.model.add_physical_group(2, top_surf, tag=5)
    gmsh.model.set_physical_name(2, 5, "TopSurface")

    occ.synchronize() # Final sync to make sure all physical group definitions are saved to the geometry

    print("\n=== Physical Groups ===") # safety check that the tags are sassigned correctly
    for d, tag in gmsh.model.getPhysicalGroups(3) + gmsh.model.getPhysicalGroups(2):
        print("Dim", d, "Tag", tag)

    gmsh.option.setNumber("Mesh.Algorithm3D", 4)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
    gmsh.model.mesh.generate(gdim)

    domain, cell_markers, facet_markers = model_to_mesh(
        gmsh.model, mesh_comm, model_rank, gdim=gdim
    )
    gmsh.finalize()

else:
    domain, cell_markers, facet_markers = model_to_mesh(None, mesh_comm, model_rank, gdim=gdim)

# Give all objects explicit names
domain.name = "dogbone_mesh"
cell_markers.name = "cell_tags"
facet_markers.name = "facet_tags"

# Define output path and save file
output_dir = "/home/ntinos/Documents/FEnics/heat equation/checkpoints"
os.makedirs(output_dir, exist_ok=True)
out_file = os.path.join(output_dir, "Dogbone3D.xdmf")

# Save mesh + tags
with io.XDMFFile(domain.comm, out_file, "w") as xdmf:
    xdmf.write_mesh(domain)  # no name argument, uses domain.name
    domain.topology.create_connectivity(fdim, gdim)
    xdmf.write_meshtags(cell_markers, domain.geometry)  # uses .name internally
    xdmf.write_meshtags(facet_markers, domain.geometry)  # uses .name internally

print("Mesh saved to:", out_file)

# ============================
# Visualize mesh with PyVista
from dolfinx.io import VTKFile
vtk = VTKFile(domain.comm, "dogbone_mesh.pvd", "w")
vtk.write_mesh(domain)
vtk.close()

import pyvista as pv

mesh = pv.read("dogbone_mesh.pvd")

plotter = pv.Plotter()
plotter.add_mesh(mesh, show_edges=False)
plotter.show(screenshot="dogbone_mesh.png")  #  Shows the window and saves the image
