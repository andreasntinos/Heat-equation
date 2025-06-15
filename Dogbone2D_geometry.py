# 04.06.2025
# The following code generates the geometry of a 2D dogbone geometry 

####_ _LIBRARIES_ _####
from mpi4py import MPI # MPI library is to carry out parallel computations
import gmsh # interface to the gmsh geometry 
import math # math.calculations, will be used for the fillet of the geometry
from dolfinx import fem, mesh, io # io is for writing XMDF files(providing input and ouptut)
from dolfinx.io.gmshio import model_to_mesh # we need this command in order to convert the gmsh model to a Dolfinx mesh
import os

gmsh.initialize()

## Specimen geometry Parameters
unit = 0.001  # the units of the spacimen dimensions are given in mm-thats why we set the unit as mm
gauge_length = 8 * unit
gauge_diameter = 5 * unit
end_diameter = 13.5 * unit
fillet_radius = 13.5 * unit
end_length_straight = 16 * unit

## Mesh Parameters
h_mesh = 0.5 * unit  # FE mesh size
# At the region of the top surface where the laser will be applied, we want the mesh to be as dense as possible
h_fine_top_edge = 0.01 * unit # Element size for the refined top gauge and fillets 
refinement_distance = 2 * unit # Distance over which refinement stops
sampling_density = 500    # gmsh parameter to control the refinement accuracy of the mesh near the top_facet (without this the maximum temperature of each)
                          # does not appear equal to the maximum temperature of the dogbone at the iteration where the laser is applied at this point

## Geometry Setup
gdim = 2 # the geometric dimension of the body
fdim = gdim - 1 # geometric dimension of the facets 
occ = gmsh.model.occ # open CASCAD engine for the geometry definition- a toolbox in gmsh that enables you to manipulate complex geometries
mesh_comm = MPI.COMM_WORLD # parallel computing communicator
model_rank = 0 # the first process of the code which will be the meshing of the geometry 

if mesh_comm.rank == model_rank: # only the main processor is responsible for the meshing fo the geometry
    gmsh.model.add("dogbone_2d") # gmsh model name
    gmsh.model.setCurrent("dogbone_2d")

    # Because of the fact that the geometry is symmetrical along the x and y axis, we can create the upper half of the dogbone and then mirror it 
    # to produce the whole geometry (use the same points but with negaive sign)

    # 1.GEOMETRY DEFINITION
    gauge_radius = gauge_diameter / 2.0
    end_radius = end_diameter / 2.0
    delta_half_width = end_radius - gauge_radius # we need this distance to define the fillet horizontal distance (dx = sqrt(dw))
    # dx_fillet is the x-projection of the fillet, the distance between the end of the gauge and the start of the end length
    dx_fillet = math.sqrt(delta_half_width * (2 * fillet_radius - delta_half_width))

    x_gauge_half = gauge_length / 2.0
    x_fillet_end = x_gauge_half + dx_fillet # x-coordinate where fillet meets the straight part of the end section
    x_outer_end = x_fillet_end + end_length_straight

    # with these lengths defined above, the 12 points of the dogbone geometry are defined
    # instead of defining the coordinates of the geometry, we add these points on the CASCADE kernel- necessary for meshing with gmsh

    p0 = occ.add_point(x_outer_end, end_radius, 0)  # top right point
    p1 = occ.add_point(x_fillet_end, end_radius, 0) # right fillet top
    p2= occ.add_point(x_gauge_half, gauge_radius, 0) # right gauge 
    # Top-Left
    p3 = occ.add_point(-x_gauge_half, gauge_radius, 0) # left gauge point
    p4 = occ.add_point(-x_fillet_end, end_radius,0 ) # 
    p5= occ.add_point(-x_outer_end, end_radius, 0)
    # Bottom-Left
    p6 = occ.add_point(-x_outer_end, -end_radius, 0)
    p7 = occ.add_point(-x_fillet_end, -end_radius, 0)
    p8= occ.add_point(-x_gauge_half, -gauge_radius, 0)
    # Bottom-Right
    p9 = occ.add_point(x_gauge_half, -gauge_radius, 0)
    p10= occ.add_point(x_fillet_end, -end_radius, 0)
    p11 = occ.add_point(x_outer_end, -end_radius, 0)
    
    #  Fillet center points (4) for the fillets
    #  4 center points for fillet creation are generated

    center_tr = occ.add_point(x_gauge_half, gauge_radius + fillet_radius, 0)
    center_tl = occ.add_point(-x_gauge_half, gauge_radius + fillet_radius, 0)
    center_bl = occ.add_point(-x_gauge_half, -gauge_radius - fillet_radius, 0)
    center_br = occ.add_point(x_gauge_half, -gauge_radius - fillet_radius, 0)

    l_top_end_r = occ.add_line(p0, p1)
    arc_tr = occ.add_circle_arc(p1, center_tr, p2)
    l_top_gauge = occ.add_line(p2, p3)
    arc_tl = occ.add_circle_arc(p3, center_tl, p4)
    l_top_end_l = occ.add_line(p4, p5)
    l_face_l = occ.add_line(p5, p6)
    l_bot_end_l = occ.add_line(p6, p7)
    arc_bl = occ.add_circle_arc(p7, center_bl, p8)
    l_bot_gauge = occ.add_line(p8, p9)
    arc_br = occ.add_circle_arc(p9, center_br, p10)
    l_bot_end_r = occ.add_line(p10, p11)
    l_face_r = occ.add_line(p11, p0)

    # Define line segments and arcs : after setting the points of the geometry, we need to define also the geometrical spaces that 
    # these points create. 

    lines = [l_top_end_r, arc_tr, l_top_gauge, arc_tl, l_top_end_l, l_face_l,
             l_bot_end_l, arc_bl, l_bot_gauge, arc_br, l_bot_end_r, l_face_r]
    
    curve_loop = occ.add_curve_loop(lines)  # then this line creates a closed loop between all the lines IDs
    surface = occ.add_plane_surface([curve_loop]) # fills the loop in order to form a surface
    occ.synchronize() # update the gmsh geometry
    
    # 2. SET ID TO THE PHYSICAL GROUPS
    gmsh.model.add_physical_group(gdim, [surface], 1, name="dogbone_surface")
    gmsh.model.add_physical_group(fdim, [l_face_l], 2, name="left_end_face")
    gmsh.model.add_physical_group(fdim, [l_face_r], 3, name="right_end_face")
    top_facet = [arc_tr, l_top_gauge, arc_tl]
    gmsh.model.add_physical_group(fdim, top_facet, 4, name="top_edge_refined")

    ## 3. MESH REFINEMENT FIELD 
    # the following lines are the definition of the refinement profile that will be used in this geometry
    gmsh.model.mesh.field.add("Distance", 1) # this line creates a distance field in gmsh with ID =1 
    gmsh.model.mesh.field.setNumbers(1, "CurvesList", top_facet) # this line tells gmsh to measure the distance from the top_facet
                                                                 # "CurvesList" is a gmsh command that says to measure distances from the curves(lines) ID's
    gmsh.model.mesh.field.setNumber(1, "Sampling", sampling_density) #controls how finely gmsh samples the curves when computing distances
 
    gmsh.model.mesh.field.add("Threshold", 2) # this line creates a distance field with ID =2
    gmsh.model.mesh.field.setNumber(2, "InField", 1) # tell distance field 2 to use the distance field 1
    gmsh.model.mesh.field.setNumber(2, "SizeMin", h_fine_top_edge) # smallest element size
    gmsh.model.mesh.field.setNumber(2, "SizeMax", h_mesh)  # largest element size
    gmsh.model.mesh.field.setNumber(2, "DistMin", 0.0) # the min_distance from the top_facet that refinement occurs, where the refinement starts
    gmsh.model.mesh.field.setNumber(2, "DistMax", refinement_distance) # the max_distance from the top_facet that refinement occurs
    gmsh.model.mesh.field.setNumber(2, "Sigmoid", True)  # the transition between size min and max becomes gradual without sharp jumps in the element size

    gmsh.model.mesh.field.setAsBackgroundMesh(2) #use field 2 as a rule to control the mesh 

    ## 4. GMSH MESHING
    # mesh the dogbone geometry with the parameters that were selected above
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h_fine_top_edge) # Global min can be the finest
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h_mesh)
    # next 3 lines are for disabling the automatic meshing options of gmsh and use only the mesh rules that we set above
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

    gmsh.model.mesh.generate(gdim) 


    # 5. CONVERT MESH TO DOLFINX AND THEN FINALIZE GMSH
    domain, cell_markers, facet_markers = io.gmshio.model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=gdim)
gmsh.finalize() 

# Give all objects explicit names before saving
domain.name = "dogbone_mesh"
cell_markers.name = "cell_tags"
facet_markers.name = "facet_tags"

# Define output path and save file
output_dir = "/home/ntinos/Documents/FEnics/heat equation/checkpoints"
os.makedirs(output_dir, exist_ok=True)
out_file = os.path.join(output_dir, "Dogbone2D.xdmf")

# Save to the folder checkpoints where all the xdmfs are
with io.XDMFFile(domain.comm, out_file, "w") as xdmf:
    xdmf.write_mesh(domain)
    domain.topology.create_connectivity(fdim, gdim)
    # Add the required domain.geometry argument back
    xdmf.write_meshtags(cell_markers, domain.geometry)
    xdmf.write_meshtags(facet_markers, domain.geometry)