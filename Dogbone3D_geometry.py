# 22.06.2025

#=================================
#           LIBRARIES            #
#=================================

from mpi4py import MPI # MPI library is to carry out parallel computations
import gmsh # interface to the gmsh geometry 
import math # math.calculations, will be used for the fillet of the geometry
from dolfinx import fem, mesh, io # io is for writing XMDF files(providing input and ouptut)
from dolfinx.io.gmshio import model_to_mesh # we need this command in order to convert the gmsh model to a Dolfinx mesh
import numpy as np
import os 

gmsh.initialize()

# All geometric parameters
unit = 0.001
gauge_length = 8 * unit
gauge_diameter = 5 * unit
end_diameter = 13.5 * unit
fillet_radius = 13.5 * unit
end_length_straight = 16 * unit

# Mesh size parameters
h_mesh = 0.8 * unit  # FE mesh size
# At the region of the top surface where the laser will be applied, we want the mesh to be as dense as possible
h_fine = 0.02 * unit # Element size for the refined top gauge and fillets 


gdim = 3 # the geometric dimension of the body
fdim = gdim - 1 # geometric dimension of the facets 
occ = gmsh.model.occ # open CASCAD engine for the geometry definition- a toolbox in gmsh that enables you to manipulate complex geometries
mesh_comm = MPI.COMM_WORLD # parallel computing communicator
model_rank = 0 # the first process of the code which will be the meshing of the geometry 

# The geometry and mesh will only be created on a single processor
if mesh_comm.rank == model_rank:
    gmsh.model.add("dogbone_3d_refined")
    gmsh.model.setCurrent("dogbone_3d_refined")
    occ = gmsh.model.occ

    # geometry definition (2D)
    gaude_radius = gauge_diameter / 2.0
    end_radius = end_diameter / 2.0
    delta_half_width = end_radius - gaude_radius
    dx_fillet = math.sqrt(delta_half_width * (2 * fillet_radius - delta_half_width))
    x_gauge_half = gauge_length / 2.0
    x_fillet_end = x_gauge_half + dx_fillet
    x_outer_end = x_fillet_end + end_length_straight

    # The difference with the 2D case is that now we do not neeed to add 1 point but only 8: create the upper part of the specimen geometry
    # and then revolve it to produce the 3D cylindrical geometry
    gpts = [
        occ.add_point(x_outer_end, 0, 0), occ.add_point(x_outer_end, end_radius, 0),
        occ.add_point(x_fillet_end, end_radius, 0), occ.add_point(x_gauge_half, gaude_radius, 0),
        occ.add_point(-x_gauge_half, gaude_radius, 0), occ.add_point(-x_fillet_end, end_radius, 0),
        occ.add_point(-x_outer_end, end_radius, 0), occ.add_point(-x_outer_end, 0, 0)
    ]
    gcenters = [occ.add_point(x_gauge_half, gaude_radius + fillet_radius, 0), occ.add_point(-x_gauge_half, gaude_radius + fillet_radius, 0)]
    
    lines = {
        "right_face": occ.add_line(gpts[0], gpts[1]), "top_straight_right_end": occ.add_line(gpts[1], gpts[2]),
        "fillet_tr": occ.add_circle_arc(gpts[2], gcenters[0], gpts[3]), "top_gauge": occ.add_line(gpts[3], gpts[4]),
        "fillet_tl": occ.add_circle_arc(gpts[4], gcenters[1], gpts[5]), "top_straight_left_end": occ.add_line(gpts[5], gpts[6]),
        "left_face": occ.add_line(gpts[6], gpts[7]), "bottom_closure": occ.add_line(gpts[7], gpts[0])
    }
    curve_loop_segments = [lines[i] for i in ["right_face", "top_straight_right_end", "fillet_tr", "top_gauge", "fillet_tl", "top_straight_left_end", "left_face", "bottom_closure"]]
    curve_loop = occ.add_curve_loop(curve_loop_segments)
    surface = occ.add_plane_surface([curve_loop])
    occ.synchronize()

    # Mesh refinement: this is an easier way to apply the refinement to the mesh by applying a constant mesh around the top lines, which becomes
    # less dense as you get far from the top surface, (the difference with the previous method is that now cant achieve a smooth transition 
    # between the mesh_size and the fine_mesh size)
    
    #refined_lines = [lines["fillet_tr"], lines["top_gauge"], lines["fillet_tl"]] #in case you want to refine the whole upper laser path
    refined_lines = [lines["fillet_tr"], lines["top_gauge"], lines["fillet_tl"]]
    gmsh.model.mesh.field.add("Constant", 1)
    gmsh.model.mesh.field.setNumber(1, "VIn", h_fine)
    gmsh.model.mesh.field.setNumbers(1, "CurvesList", refined_lines)
    gmsh.model.mesh.field.add("Min", 2)
    gmsh.model.mesh.field.setNumbers(2, "FieldsList", [1])
    gmsh.model.mesh.field.setAsBackgroundMesh(2)
    gmsh.option.setNumber("Mesh.MeshSizeMax", h_mesh)

    # Revolve geometry 
    revolved_entities = occ.revolve([(2, surface)], 0, 0, 0, 1, 0, 0, 2 * math.pi) # this line says to find the 2D shape that has the ID
                                                                                   # surface and then revolve it around for 2*pi
    occ.synchronize()

    volume_tag = next((entity[1] for entity in revolved_entities if entity[0] == 3), None) 

    # Tag the volume 
    gmsh.model.add_physical_group(3, [volume_tag], tag=1) # 3 is the dimension, volume_tag is the variable that holds the ID and then
                                                          # tag=1 is the specific ID for the whole cylindrical volume
    gmsh.model.set_physical_name(3, 1, "DogboneVolume") 

    # Get all boundary surfaces of the volume: the following line searches for the boundaries of the entity with dimension 3 and the tag stored
    # in the volume_tag variable
    boundary_surfaces = gmsh.model.get_boundary([(3, volume_tag)], oriented=False, combined=False) 

    # Detect and tag left and right flat end surfaces
    tol = 1e-6
    left_face_entities, right_face_entities, laser_face_entities, top_surface_tags = [], [], [], []

    # The following loop is what defines the left and right boundaries. We have defined boundary_surfaces that returned all the 2D boundaries of the volume
    # Then we scan this result checking all the dimensions and the tags with the "gmsh.model.occ.get_center_of_mass" that returns the center of mass
    # of each geometry. IF the center of mass is equal to the point in the end of the -length by a tolerance of tol, then this 2D geometry
    # is the left surface, ELSE it is the right side. IF the center of mass in the x direction is smaller than the half of gauge length
    # then it is the top surface of the specimen 

    for dim, tag in boundary_surfaces:
        com = gmsh.model.occ.get_center_of_mass(dim, tag)
        if math.isclose(com[0], -x_outer_end, abs_tol=tol):
            left_face_entities.append(tag)
        elif math.isclose(com[0], x_outer_end, abs_tol=tol):
            right_face_entities.append(tag)
        elif abs(com[0]) < x_fillet_end:
            # This correctly identifies only the top surfaces (gauge + fillets) for losses
            top_surface_tags.append(tag)
        else:
            laser_face_entities.append(tag)

    gmsh.model.add_physical_group(2, left_face_entities, tag=2)
    gmsh.model.set_physical_name(2, 2, "LeftEnd")

    gmsh.model.add_physical_group(2, right_face_entities, tag=3)
    gmsh.model.set_physical_name(2, 3, "RightEnd")

    gmsh.model.add_physical_group(2, laser_face_entities, tag=4)
    gmsh.model.set_physical_name(2, 4, "LaserSurface")

    gmsh.model.add_physical_group(2, top_surface_tags, tag=5)
    gmsh.model.set_physical_name(2, 5, "TopSurface")

    occ.synchronize()

    print("\n=== Physical Groups ===")
   
    for dim in [2, 3]:
        groups = gmsh.model.getPhysicalGroups(dim)
        for (d, tag) in groups:
           name = gmsh.model.getPhysicalName(d, tag)
           entities = gmsh.model.getEntitiesForPhysicalGroup(d, tag)
           print(f"{name} (dim={d}, tag={tag}) has entities: {entities}")
    
    gmsh.model.mesh.generate(gdim) # generate mesh 
    # 5. CONVERT MESH TO DOLFINX AND FINALIZE GMSH
    domain, cell_markers, facet_markers = model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=gdim)
gmsh.finalize()

# Naming
domain.name = "dogbone_mesh"
cell_markers.name = "cell_tags"
facet_markers.name = "facet_tags"
print("Facet markers:", np.unique(facet_markers.values))

# Save path
output_dir = "/home/ntinos/Documents/FEnics/heat equation/checkpoints"
os.makedirs(output_dir, exist_ok=True)
out_file = os.path.join(output_dir, "Dogbone3D.xdmf")

# Export to XDMF
with io.XDMFFile(domain.comm, out_file, "w") as xdmf:
    xdmf.write_mesh(domain)
    domain.topology.create_connectivity(fdim, gdim)
    xdmf.write_meshtags(facet_markers, domain.geometry)

