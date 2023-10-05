import numpy as np
import cv2
import math
import random
import sys
from numba import jit
from pythonabm import Simulation, record_time, check_direct
from shapely.geometry import Polygon

#overlap is a boolean$

def get_gravity_forces(number_cells, locations, center, well_rad, net_forces):
    for index in range(number_cells):
        new_loc = locations[index] - center
        # net_forces[index] = -1 * (new_loc / well_rad) * np.sqrt((np.linalg.norm(new_loc) / well_rad) ** 2)
        new_loc_sum = new_loc[0] ** 2 + new_loc[1] ** 2
        net_forces[index] = -1 * (new_loc / well_rad) * np.sqrt(1 - new_loc_sum / well_rad ** 2)
    return net_forces


def get_neighbor_forces(edge_forces, num_agents, contacts, locations, edge_locations, radii, center, u_rep=10):
    # Construct polygons:
    poly_list = getPolygons(num_agents, edge_locations, center)
    for i in range(num_agents):
        vecs, dist, overlap, interaction = check_overlap_poly(locations[i], np.vstack((locations[:i], locations[i+1:])), 
                                                              poly_list[i], poly_list[:i] + poly_list[i+1:], center, radii[i])
        #vecs, dist, overlap, interaction = check_overlap(locations[i], locations, center, radii[i] * 2, radii[i] * 3.2)
        net_force_repulsion = u_rep * np.sum((vecs / dist * overlap), axis=0)
        magnitude_repulsion = np.sqrt(np.sum(np.square(net_force_repulsion)))
        #magnitude = np.sqrt(np.sum(np.square(net_force)))
        net_force_attraction = np.sum((vecs[interaction] / dist[interaction].reshape((len(dist[interaction]), 1))),
                                      axis=0)
        magnitude_attraction = np.sqrt(np.sum(np.square(net_force_attraction)))
        if magnitude_attraction == 0:
            edge_forces[i] = 0
        else:
            edge_forces[i] = -1 * net_force_repulsion + net_force_attraction
            edge_forces[i] = edge_forces[i] / (magnitude_repulsion + magnitude_attraction)
        overlap[overlap > 0] = 1
        contacts[i] = np.sum(overlap)
    return edge_forces, contacts
#weigh cells more for attraction

def getPolygons(num_cells, edge_locations, center):
    poly_list = []
    for i in range(num_cells):
        corners = []
        for j in range(edge_locations.shape[2]):
            corners.append((edge_locations[i, 0, j] - center[0], edge_locations[i, 1, j] - center[1]))
        poly_list.append(Polygon(corners))
    return poly_list

def check_overlap(cell1, locations, center, overlap_distance, attraction_distance):
    '''
    check overlap of cells given 2 sets of vertices.
    Return 1 if true, 0 if false
    '''
    cell_1_loc = cell1 - center
    vecs = locations - center - cell_1_loc
    dist = np.sqrt(np.sum(np.square(vecs), axis=1))
    overlap = (dist < overlap_distance) * (dist > 0)
    interaction = (dist > overlap_distance) * (dist <= attraction_distance)
    return vecs, dist, overlap, interaction
    #overlap is the boolean

def check_overlap_poly(cell1, locations, current_poly, poly_list, center, attraction_distance):
    intersect_area = []
    overlap = []
    for poly in poly_list:
        area = current_poly.intersection(poly).area
        intersect_area.append((area, area))
        overlap.append(~current_poly.intersects(poly))
    intersect_area = np.array(intersect_area)
    overlap = np.array(overlap)
    cell_1_loc = cell1 - center
    vecs = locations - center - cell_1_loc
    dist = np.sqrt(np.sum(np.square(vecs), axis=1))
    interaction = (dist <= attraction_distance) * (overlap)
    return vecs, dist.reshape(len(poly_list), 1), intersect_area, interaction

def seed_cells(num_agents, center, radius):
    theta = 2 * np.pi * np.random.rand(num_agents).reshape(num_agents, 1)
    rad = radius * np.sqrt(np.random.rand(num_agents)).reshape(num_agents, 1)
    x = rad * np.cos(theta) + center[0]
    y = rad * np.sin(theta) + center[1]
    #z = np.zeros((num_agents, 1)) + center[2]
    #z = np.zeros((num_agents, 1)) + 5
    #locations = np.hstack((x, y, z))
    locations = np.hstack((x,y))
    return locations

def update_corners(cell_locations, corner_locations, orientations, major, minor):
    translation_matrix = np.array(([major, minor], [major, -minor], [-major, -minor], [-major, minor]))
    for i in range(corner_locations.shape[2]):
        corner_locations[:,:,i] = cell_locations[:,:] + translation_matrix[i].transpose()
        for j in range(corner_locations.shape[0]):
            corner_locations[j,:,i] = rotate(corner_locations[j,:,i], orientations[j], cell_locations[j])
    return corner_locations

def rotate(point, theta, center):
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c))).transpose()
    return np.matmul(point-center, R) + center

def RandomWalkXYZ(N, d):
    random_walk = np.random.uniform(low=-0.5, high=0.5, size=(N,d))
    return np.cumsum(random_walk, axis=0)

def seed_cells_sphere(center, W, N=50, d=2):
    """
    :param center: center of sphere. Likely that we would want this to be the center of the simulation space.
    :param W: number of cells
    :param N: number of "steps" to take of random walk. Related to how dense we want our sphere to be
    :param d: the dimensions of the model. d=3 is 3d, d=2 is 2d
    """
    endpoints = np.zeros((W, d))
    for i in range(W):
        endpoints[i] = RandomWalkXYZ(N, d)[-1, :]
    return endpoints + center



def area_constraint(area, major_axis):
    """
    Cells can only have a certain volume, so if the cell is elongating, the minor axis should be decreasing
    to preserve a fixed cell area.
    """
    return area/(np.pi * major_axis)


class NeuralRosette(Simulation):
    """ This class defines the necessary behavior for any Simulation
        subclass.
    """
    def __init__(self):
        # hold simulation name (will be overridden)
        self.name = "Trial"

        # hold the current number of agents and the step to begin at (updated by continuation mode)
        self.number_agents = 0
        self.current_step = 0

        # hold the real time start of the step in seconds and the total time for select methods
        self.step_start = 0
        self.method_times = dict()

        # hold the names of the agent arrays and the names of any graphs (each agent is a node)
        self.array_names = list()
        self.graph_names = list()

        # hold bounds for specifying types of agents that vary in initial values
        self.agent_types = dict()

        # default values which can be updated in the subclass
        self.num_to_start = 112
        self.cuda = False
        self.end_step = 30
        self.size = [300, 300, 0]
        self.center = [150, 150]
        self.output_values = True
        self.output_images = True
        self.image_quality = 900
        self.video_quality = 1000
        self.fps = 5
        self.tpb = 4    # blocks per grid for CUDA neighbor search, use higher number if performance is slow

        self.diff_color = np.array([255, 50, 50], dtype=int) #red
        self.transition_color = np.array([191, 64, 191], dtype=int) #purple
        self.stem_color = np.array([50, 50, 255], dtype=int) #blue
        self.velocity = .05
        self.cell_interaction_rad = 3.2
        self.min_cell_size = 5
        self.max_cell_size = 10
        self.cell_volume = np.pi * self.min_cell_size ** 2

    def setup(self):
        """ Initialize the simulation prior to running the steps. Must
            be overridden.
        """

        # (S)tem cells, (T)ransition cells, (D)ifferentiated Cells
        self.add_agents(self.num_to_start, agent_type="S")
        self.add_agents(0, agent_type="T")
        self.add_agents(0, agent_type="D")

        self.indicate_arrays("locations", "major_axis", "minor_axis", "colors", "cell_type", "division_set", "division_threshold",
                             "diff_set", "diff_threshold", "orientations", "contacts")
        #self.locations = seed_cells_square(self.num_to_start, 1000, 2)
        #centerr = [self.size[1]/2, self.size[2]/2]
        self.locations = seed_cells_sphere(self.center, self.num_to_start, N=10000)
        self.locations = np.where(self.locations > 300, 300, self.locations)
        self.locations = np.where(self.locations < 0, 0, self.locations)

        self.locations_corners = np.zeros(self.locations.shape + (4,))

        self.cell_type = self.agent_array(dtype=int, initial={"S": lambda: 0,
                                                              "T": lambda: 1,
                                                              "D": lambda: 2})
        # Currently making the cells entirely the same size, but we can apply variation depending on what is needed
        self.major_axis = self.agent_array(initial={"S": lambda: self.min_cell_size, 
                                                    "T": lambda: self.min_cell_size,
                                                    "D": lambda: self.min_cell_size})
        self.minor_axis = self.agent_array(initial={"S": lambda: self.min_cell_size, 
                                                    "T": lambda: self.min_cell_size,
                                                    "D": lambda: self.min_cell_size})
        self.orientations = self.agent_array(initial= {"S": lambda: np.random.uniform(0, 2*math.pi),
                                                       "T": lambda: np.random.uniform(0, 2*math.pi),
                                                       "D": lambda: np.random.uniform(0, 2*math.pi)})
        
        self.locations_corners = update_corners(self.locations, self.locations_corners, 
                                                self.orientations, self.major_axis, self.minor_axis)

        self.colors = self.agent_array(dtype=int, vector=3, initial={"S": lambda: self.stem_color,
                                                              "T": lambda: self.transition_color,
                                                              "D": lambda: self.diff_color})

        self.division_set = self.agent_array(initial={"S": lambda: np.random.uniform(0, 19),
                                                "T": lambda: 0,
                                                "D": lambda: 0})
        self.diff_set = self.agent_array(initial={"S": lambda: np.random.uniform(0, 1),
                                                "T": lambda: 0,
                                                "D": lambda: 0})
        self.division_threshold = self.agent_array(initial={"S": lambda: 19,
                                                            "T": lambda: 1000,
                                                            "D": lambda: 51})
        self.diff_threshold = self.agent_array(initial={"S": lambda: 1,
                                                        "T": lambda: -1,
                                                        "D": lambda: -1})
        self.contacts = self.agent_array(initial=lambda:0)
        self.indicate_graphs("neighbor_graph")
        self.neighbor_graph = self.agent_graph()
        for _ in range(80):
            self.move()
        self.step_values()
        self.step_image()



    def step(self):
        """ Specify which methods are called during the simulation step.
            Must be overridden.
        """
    # Difference in timescales
        self.move()
        self.stretch_cells()
        #update up here
        self.step_values()
        self.step_image()


    def end(self):
        """ Specify any methods to be called after all the simulation
            steps have run. Can be overridden.
        """
        self.create_video()

    def info(self):
        """ Prints out info about the simulation.
        """
        # current step and number of agents
        print("Step: " + str(self.current_step))
        print("Number of agents: " + str(self.number_agents))

    def mark_to_hatch(self, index):
        """ Mark the corresponding index of the array with True to
            indicate that the agent should hatch a new agent.

            :param index: The unique index of an agent.
            :type index: int
        """
        self.hatching[index] = True

    def mark_to_remove(self, index):
        """ Mark the corresponding index of the array with True to
            indicate that the agent should be removed.

            :param index: The unique index of an agent.
            :type index: int
        """
        self.removing[index] = True


    @record_time
    def update_populations(self):
        """ Adds/removes agents to/from the simulation by adding/removing
            indices from the cell arrays and any graphs.
        """
        # get indices of hatching/dying agents with Boolean mask
        add_indices = np.arange(self.number_agents)[self.hatching]
        remove_indices = np.arange(self.number_agents)[self.removing]

        # count how many added/removed agents
        num_added = len(add_indices)
        num_removed = len(remove_indices)

        # go through each agent array name
        for name in self.array_names:
            # Split intracellular cAMP between daughter cells.

            # copy the indices of the agent array data for the hatching agents
            copies = self.__dict__[name][add_indices]

            # add/remove agent data to/from the arrays
            self.__dict__[name] = np.concatenate((self.__dict__[name], copies), axis=0)
            self.__dict__[name] = np.delete(self.__dict__[name], remove_indices, axis=0)

        # go through each graph name
        for graph_name in self.graph_names:
            # add/remove vertices from the graph
            self.__dict__[graph_name].add_vertices(num_added)
            self.__dict__[graph_name].delete_vertices(remove_indices)

        # change total number of agents and print info to terminal
        self.number_agents += num_added - num_removed
        print("\tAdded " + str(num_added) + " agents")
        print("\tRemoved " + str(num_removed) + " agents")

        # clear the hatching/removing arrays for the next step
        self.hatching[:] = False
        self.removing[:] = False

    def stretch_cells(self):
        """
        Probably need a more elaborate method of stretching cells, this is probably the bulk of the modeling
        in this process.
        
        Currently cells will just stretch at random in a random direction. Need to determine what makes the cells stretch
        and what determines what direction they stretch in.
        """
        # I decided arbitrarily that a cell can stretch at most 10% of its starting cell size per timestep
        self.major_axis += np.random.uniform(-self.min_cell_size * 0.1, self.min_cell_size * 0.1, size=self.number_agents)
        # Constraint: major axis can't be < 5 um (min cell size) and the cells can't stretch more than the max cell size.
        self.major_axis = np.where(self.major_axis > self.max_cell_size, self.max_cell_size, self.major_axis)
        self.major_axis = np.where(self.major_axis < self.min_cell_size, self.min_cell_size, self.major_axis)
        
        self.minor_axis = area_constraint(self.cell_volume, self.major_axis)

        # I decided arbitrarily that a cell can rotate at most 10 degrees per timestep
        self.orientations += np.random.uniform(-math.pi/18, math.pi/18, size=self.number_agents)
        self.locations_corners = update_corners(self.locations, self.locations_corners, 
                                                self.orientations, self.major_axis, self.minor_axis)
        pass

    def move(self):
        #image_center = [self.size[1]/2, self.size[2]/2]
        neighbor_forces = np.zeros((self.number_agents, 2))
        neighbor_forces, self.contacts = get_neighbor_forces(neighbor_forces, self.number_agents, self.contacts,
                                             self.locations, self.locations_corners, self.radii, self.center)
        # gravity_forces = np.zeros((self.number_agents, 2))
        # gravity_forces = get_gravity_forces(self.number_agents, self.locations, self.center, 300, gravity_forces)
        #average_position = np.mean(self.locations, axis = 0)
        #direction = self.locations - image_center
        #norm_direction = direction/np.linalg.norm(direction, axis=1, keepdims=True)
        #neighbor_forces = (neighbor_forces * .5) + 1 * gravity_forces
        noise_vector = np.random.normal(0, 1, (self.number_agents, 2))

        # (ri/r)sqrt(S

        time_step = 1

        self.locations += time_step * (self.radii.reshape((self.number_agents, 1))) * self.velocity * (
                    neighbor_forces + noise_vector)
        #check that the new location is within the space, otherwise use boundary values
        self.locations = np.where(self.locations > 300, 300, self.locations)
        self.locations = np.where(self.locations < 0, 0, self.locations)


    def step_image(self, background=(0, 0, 0), origin_bottom=True):
        """ Creates an image of the simulation space.

            :param background: The 0-255 RGB color of the image background.
            :param origin_bottom: If true, the origin will be on the bottom, left of the image.
            :type background: tuple
            :type origin_bottom: bool
        """
        # only continue if outputting images
        if self.output_images:
            # get path and make sure directory exists
            check_direct(self.images_path)

            # get the size of the array used for imaging in addition to the scaling factor
            x_size = self.image_quality
            scale = x_size / self.size[0]
            y_size = math.ceil(scale * self.size[1])

            # create the agent space background image and apply background color
            image = np.zeros((y_size, x_size, 3), dtype=np.uint8)
            background = (background[2], background[1], background[0])
            image[:, :] = background


            # go through all of the agents
            for index in range(self.number_agents):
                # get xy coordinates, the axis lengths, and color of agent
                x, y = int(scale * self.locations[index][0]), int(scale * self.locations[index][1])
                corners = []
                for i in range(self.locations_corners.shape[2]):
                    corners.append([int(scale * self.locations_corners[index, 0, i]), int(scale*self.locations_corners[index, 1, i])])
                corners = np.array(corners, dtype=int)

                #major, minor = int(scale * self.major_axis[index]), int(scale * self.minor_axis[index])
                color = (int(self.colors[index][2]), int(self.colors[index][1]), int(self.colors[index][0]))
                #angle = self.orientations[index] * 180 / np.pi
                # draw the agent and a black outline to distinguish overlapping agents
                # image = cv2.ellipse(image, (x, y), (major, minor), angle, 0, 360, color, -1)
                # image = cv2.ellipse(image, (x, y), (major, minor), angle, 0, 360, (0, 0, 0), 1)

                image = cv2.fillPoly(image, pts=[corners], color=color)
                image=cv2.fillPoly(image, pts=[corners[0:2]], color=(255, 255,255))


                # add polarity into axes

            # if the origin should be bottom-left flip it, otherwise it will be top-left
            if origin_bottom:
                image = cv2.flip(image, 0)

            # save the image as a PNG
            image_compression = 4  # image compression of png (0: no compression, ..., 9: max compression)
            file_name = f"{self.name}_image_{self.current_step}.png"
            cv2.imwrite(self.images_path + file_name, image, [cv2.IMWRITE_PNG_COMPRESSION, image_compression])


if __name__ == '__main__':
    sim = NeuralRosette()
    if sys.platform == 'win32':
        sim.start("C:\\Users\\ajin40\\Documents\\sim_outputs\\neural_rosettes\\outputs")
    elif sys.platform == 'darwin':
        sim.start("/Users/andrew/Projects/sim_outputs/neural_rosettes/outputs")
    elif sys.platform =='linux':
        sim.start('/home/ajin40/models/model_outputs')
    else:
        print('I did not plan for another system platform... exiting...')
