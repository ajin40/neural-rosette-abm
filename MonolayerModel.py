import numpy as np
from numba import jit, prange
from pythonabm.simulation import Simulation, record_time
import pythonabm.backend as backend
import scipy.spatial.distance
import cv2
import sys
#

@jit(nopython=True, parallel=True)
def get_neighbor_forces(number_edges, edges, edge_forces, locations, center, types, radius, u_adhesion, u_repulsion, r_e):
    for index in prange(number_edges):
        # get indices of cells in edge
        cell_1 = edges[index][0]
        cell_2 = edges[index][1]
        # get cell positions
        cell_1_loc = locations[cell_1] - center
        cell_2_loc = locations[cell_2] - center

        # get new location position
        vec = cell_2_loc - cell_1_loc
        dist2 = vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2

        # based on the distance apply force differently
        if dist2 == 0:
            edge_forces[index][0] = 0
            edge_forces[index][1] = 0
        else:
            dist = dist2 ** (1/2)
            if 0 < dist < (2 * radius):
                edge_forces[index][0] = -1 * u_repulsion * (vec / dist)
                edge_forces[index][1] = 1 * u_repulsion * (vec / dist)
            else:
                # get the cell type
                # cell_1_type = types[cell_1]
                # cell_2_type = types[cell_2]
                # get value prior to applying type specific adhesion const
                value = (dist - r_e) * (vec / dist)
                edge_forces[index][0] =  u_adhesion * value
                edge_forces[index][1] = -1 * u_adhesion * value
    return edge_forces


@jit(nopython=True, parallel=True)
def convert_edge_forces(number_edges, edges, edge_forces, neighbor_forces):
    for index in prange(number_edges):
        # get indices of cells in edge
        cell_1 = edges[index][0]
        cell_2 = edges[index][1]

        neighbor_forces[cell_1] += edge_forces[index][0]
        neighbor_forces[cell_2] += edge_forces[index][1]

    return neighbor_forces

def seed_cells_square(num_agents, radius, edge=2):
    locations = (radius - 2 * edge) * np.random.rand(num_agents, 3) + edge
    locations[:,2] = 0
    return locations


class MonolayerModel(Simulation):
    """ This class inherits the Simulation class allowing it to run a
        simulation with the proper functionality.
    """
    def __init__(self, model_params):
        # initialize the Simulation object
        Simulation.__init__(self)

        self.default_parameters = {
            "size": [1, 1, 0],
            "dimension": 3,
            "well_rad": 30,
            "output_values": True,
            "output_images": True,
            "image_quality": 900,
            "video_quality": 900,
            "fps": 12,
            "E_cell_rad": 0.5,
            "M_cell_rad": 0.5,
            "cell_deformation": 0,
            "velocity": 0.05,
            "initial_seed_ratio": 0.5,
            "cell_interaction_rad": 2.4,
            "cell_rad": 0.5,
            "replication_type": None,
        }
        self.model_parameters(self.default_parameters)
        self.model_parameters(model_params)
        self.model_params = model_params

        # aba/dox/cho ratio
        self.E_color = np.array([255, 255, 255], dtype=int) #blue
        self.M_color = np.array([0, 0, 255], dtype=int)

        self.initial_seed_rad = self.well_rad * self.initial_seed_ratio
        self.dim = np.asarray(self.size)
        self.size = self.dim * self.well_rad
        self.center = np.array([self.size[0] / 2, self.size[1] / 2, 0])
        #self.dt = self.dx2 * self.dy2 / (2 * self.inducer_D * (self.dx2 + self.dy2))


    def setup(self):
        """ Overrides the setup() method from the Simulation class.
        """

        # initial seeded cell populations (M=mesoderm, E=endoderm, ME=mesendoderm)
        num_M = int(self.num_to_start * (1 - self.E_ratio))
        num_E = int(self.num_to_start * self.E_ratio)

        # add agents to the simulation
        self.add_agents(num_E, agent_type="E")
        self.add_agents(num_M, agent_type="M")

        # indicate agent arrays and create the arrays with initial conditions
        self.indicate_arrays("locations", "radii", "colors", "cell_type", "division_set", "div_thresh")

        # generate random locations for cells
        #self.locations = seed_cells(self.number_agents, self.center, self.initial_seed_rad)
        self.locations = seed_cells_square(self.number_agents, self.well_rad)
        self.vecs = np.zeros((self.number_agents, self.number_agents))
        self.dist = np.zeros((self.number_agents, self.number_agents))
        self.overlap = np.zeros((self.number_agents, self.number_agents))

        # Define cell types, 0 is (M)esoderm, 1 is (E)ndoderm.
        self.cell_type = self.agent_array(dtype=int, initial={"M": lambda: 0,
                                                              "E": lambda: 1})

        self.radii = self.agent_array(initial={"M": lambda: self.M_cell_rad,
                                               "E": lambda: self.E_cell_rad})

        self.colors = self.agent_array(dtype=int, vector=3, initial={"M": lambda: self.M_color,
                                                                     "E": lambda: self.E_color})

        # setting contact free division times in hours:
        # Not used in model for now
        self.div_thresh = self.agent_array(initial={"M": lambda: 18, "E": lambda: 18})
        self.division_set = self.agent_array(initial={"M": lambda: np.random.uniform(0, 18, 1),
                                                      "E": lambda: np.random.uniform(0, 18, 1)})

        #indicate and create graphs for identifying neighbors
        self.indicate_graphs("neighbor_graph")
        self.neighbor_graph = self.agent_graph()

        # save parameters to text file
        self.save_params(self.model_params)

        #Make sure cells aren't touching
        for i in range(100):
            
            self.get_neighbors(self.neighbor_graph, self.cell_interaction_rad * self.cell_rad)
            self.move_parallel()
        self.update_states(0)
        # record initial values
        self.step_values()
        #self.step_image(morphogen_background=self.activator_field)
        self.step_image()

    def step(self):
        """ Overrides the step() method from the Simulation class.
        """
        # preform subset force and RD calculations
        for i in range(self.sub_ts):
            # get all neighbors within threshold (1.6 * diameter)
            self.get_neighbors(self.neighbor_graph, self.cell_interaction_rad * self.cell_rad)
            # move the cells and track total repulsion vs adhesion forces
            self.move_parallel()
        self.update_states(0.5)

        # add/remove agents from the simulation
        self.update_populations()
        print(f'Num_M: {len(np.argwhere(self.cell_type == 1))}, Num_E: {len(np.argwhere(self.cell_type == 0))}')

        self.step_values()
        #self.step_image(morphogen_background=self.activator_field)
        self.step_image()
        self.temp()
        self.data()

    @record_time
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
            backend.check_direct(self.images_path)

            # get the size of the array used for imaging in addition to the scaling factor
            x_size = self.image_quality
            scale = x_size / self.size[0]
            y_size = int(np.ceil(scale * self.size[1]))

            # create the agent space background image and apply background color
            image = np.zeros((y_size, x_size, 3), dtype=np.uint8)
            background = (background[2], background[1], background[0])
            image[:, :] = background

            # go through all of the agents
            for index in range(self.number_agents):
                # get xy coordinates, the axis lengths, and color of agent
                x, y = int(scale * self.locations[index][0]), int(scale * self.locations[index][1])
                major, minor = int(scale * self.radii[index]), int(scale * self.radii[index])
                color = (int(self.colors[index][2]), int(self.colors[index][1]), int(self.colors[index][0]))

                # draw the agent and a black outline to distinguish overlapping agents
                image = cv2.ellipse(image, (x, y), (major, minor), 0, 0, 360, color, -1)
                image = cv2.ellipse(image, (x, y), (major, minor), 0, 0, 360, (0,0,0), 1)

            # if the origin should be bottom-left flip it, otherwise it will be top-left
            if origin_bottom:
                image = cv2.flip(image, 0)

            # save the image as a PNG
            image_compression = 4  # image compression of png (0: no compression, ..., 9: max compression)
            file_name = f"{self.name}_image_{self.current_step}.png"
            cv2.imwrite(self.images_path + file_name, image, [cv2.IMWRITE_PNG_COMPRESSION, image_compression])

    def end(self):
        """ Overrides the end() method from the Simulation class.
        """
        self.step_values()
        self.step_image()
        self.create_video()

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
            # copy the indices of the agent array data for the hatching agents
            copies = self.__dict__[name][add_indices]

            # add indices to the arrays
            self.__dict__[name] = np.concatenate((self.__dict__[name], copies), axis=0)

            # if locations array
            if name == "locations":
                # go through the number of cells added
                for i in range(num_added):
                    # get mother and daughter indices
                    mother = add_indices[i]
                    daughter = self.number_agents + i

                    # move distance of radius in random direction
                    direction = np.random.uniform(-1, 1, self.dimension)
                    vec = self.radii[i] * direction/np.linalg.norm(direction)

                    self.__dict__[name][mother] += vec
                    self.__dict__[name][daughter] -= vec

            # reset division time
            if name == "division_set":
                # go through the number of cells added
                for i in range(num_added):
                    # get mother and daughter indices
                    mother = add_indices[i]
                    daughter = self.number_agents + i

                    # set division counter to zero
                    self.__dict__[name][mother] = 0
                    self.__dict__[name][daughter] = 0

            # set new division threshold
            if name == "div_thresh":
                # go through the number of cells added
                for i in range(num_added):
                    # get daughter index
                    daughter = self.number_agents + i

                    # set division threshold based on cell type
                    self.__dict__[name][daughter] = self.__dict__[name][mother]

            # remove indices from the arrays
            self.__dict__[name] = np.delete(self.__dict__[name], remove_indices, axis=0)

        # go through each graph name
        for graph_name in self.graph_names:
            # add/remove vertices from the graph
            self.__dict__[graph_name].add_vertices(num_added)
            self.__dict__[graph_name].delete_vertices(remove_indices)

        # change total number of agents and print info to terminal
        self.number_agents += num_added
        # print("\tAdded " + str(num_added) + " agents")
        # print("\tRemoved " + str(num_removed) + " agents")

        # clear the hatching/removing arrays for the next step
        self.hatching[:] = False
        self.removing[:] = False

    @record_time
    def move_parallel(self):
        edges = np.asarray(self.neighbor_graph.get_edgelist())
        num_edges = len(edges)
        edge_forces = np.zeros((num_edges, 2, self.dimension))
        neighbor_forces = np.zeros((self.number_agents, self.dimension))
        if num_edges > 0:
            # get adhesive/repulsive forces from neighbors and gravity forces
            edge_forces = get_neighbor_forces(num_edges, edges, edge_forces, self.locations, self.center, self.cell_type,
                                            self.cell_rad, u_adhesion=self.u_adhesion, u_repulsion=self.u_repulsion, r_e = 2.02 * self.cell_rad)
            neighbor_forces = convert_edge_forces(num_edges, edges, edge_forces, neighbor_forces)
        noise_vector = self.alpha * np.random.uniform(-1, 1, (self.number_agents, self.dimension))
        for i in range(self.number_agents):
            vec = neighbor_forces[i]
            sum = vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2
            if sum != 0:
                neighbor_forces[i] = neighbor_forces[i] / (sum ** (1/2))
            else:
                neighbor_forces[i] = 0
        self.locations += (2 * self.radii.reshape((self.number_agents, 1))) * self.velocity * (neighbor_forces + noise_vector)
        # check that the new location is within the space, otherwise use boundary values
        self.locations = np.where(self.locations > self.well_rad, self.well_rad, self.locations)
        self.locations = np.where(self.locations < 0, 0, self.locations)

    #Unused..
    @record_time
    def update_states(self, ts):
        """ If the agent meets criteria, hatch a new agent.
        """
        # increase division counter by time step for all agents
        #self.division_set += ts

        #go through all agents marking for division if over the threshold
        
        self.division_set[:] += ts
        for index in range(self.number_agents):
            num_neighbors = sum(self.neighbor_graph.get_adjacency()[index])
            if self.division_set[index] > self.div_thresh[index]:
                # 6 is the maximum number of cells that can surround a cell
                #if np.sum(adjacency_matrix[:, index]) < 5:
                if num_neighbors < 6:
                    self.mark_to_hatch(index)
            self.colors[index] =np.array([min(max(0, (6-num_neighbors)/6) * 255, 255), min(max(0, (6-num_neighbors)/6) * 255, 255), 255], dtype=int) 

    @classmethod
    def simulation_mode_0(cls, name, output_dir):
        """ Creates a new brand new simulation and runs it through
            all defined steps.z
        """
        # make simulation instance, update name, and add paths
        sim.name = name
        sim.set_paths(output_dir)

        # set up the simulation agents and run the simulation
        sim.full_setup()
        sim.run_simulation()

    def save_params(self, params):
        """ Add the instance variables to the Simulation object based
            on the keys and values from a YAML file.
        """

        # iterate through the keys adding each instance variable
        with open(self.main_path + "parameters.txt", "w") as parameters:
            for key in list(params.keys()):
                parameters.write(f"{key}: {params[key]}\n")
        parameters.close()


    def model_parameters(self, model_params):
        """ Add the instance variables to the Simulation object based
            on the keys and values from a YAML file.

            :param model_params: List of all parameters.
            :type model_params: dictionary
        """

        for key in list(model_params.keys()):
            self.__dict__[key] = model_params[key]

    def add_agents(self, number, agent_type=None):
        """ Adds number of agents to the simulation.

            :param number_agents: The current number of agents in the simulation.
            :type number_agents: int
        """
        # determine bounds for array slice and increase total agents
        begin = self.number_agents
        self.number_agents += number

        # if an agent type identifier is passed, set key value to tuple of the array slice
        if agent_type is not None:
            self.agent_types[agent_type] = (begin, self.number_agents - 1)

        # go through each agent array, extending them to the new size
        for array_name in self.array_names:
            # get shape of new agent array
            shape = np.array(self.__dict__[array_name].shape)
            shape[0] = number

            # depending on array, create new array to append to the end of the old array
            if array_name == "locations":
                array = np.random.rand(number, len(self.size)) * self.size
            elif array_name == "radii":
                array = 5 * np.ones(number)
            elif array_name == "colors":
                array = np.full(shape, np.array([0, 0, 255]), dtype=int)
            else:
                # get data type and create array
                dtype = self.__dict__[array_name].dtype
                if dtype in (str, tuple, object):
                    array = np.empty(shape, dtype=object)
                else:
                    array = np.zeros(shape, dtype=dtype)

            # add array to existing agent arrays
            self.__dict__[array_name] = np.concatenate((self.__dict__[array_name], array), axis=0)

        # go through each agent graph, adding number agents to it
        for graph_name in self.graph_names:
            self.__dict__[graph_name].add_vertices(number)

if __name__ == "__main__":
    if sys.platform == 'win32':
        model_params = {
            "num_to_start": 400,
            "alpha": 0.5,
            "end_step": 36,
            "sub_ts": 1,
            "E_ratio": 1,
            "u_adhesion": 1,
            "u_repulsion": 100,
            "PACE": False,
            "cuda": True
        }
        sim = MonolayerModel(model_params)
        sim.start("C:\\Users\\ajin40\\Documents\\sim_outputs\\neural-rosette\\outputs")
    elif sys.platform == 'darwin':
        model_params = {
            "num_to_start": 1000,
            "alpha": 0.05,
            "end_step": 36,
            "sub_ts": 1,
            "E_ratio": 1,
            "u_adhesion": 1,
            "u_repulsion": 100,
            "PACE": False,
            "cuda": False
        }
        sim = MonolayerModel(model_params)
        sim.start("/Users/andrew/Projects/sim_outputs/neural-rosette/outputs")
    elif sys.platform =='linux':
        model_params = {
            "num_to_start": 1000,
            "alpha": 0.5,
            "end_step": 36,
            "sub_ts": 1,
            "E_ratio": 1,
            "u_adhesion": 1,
            "u_repulsion": 100,
            "PACE": False,
            "cuda": False
        }
        sim = MonolayerModel(model_params)
        sim.start('/home/ajin40/models/model_outputs', model_params)
    else:
        print('I did not plan for another system platform... exiting...')

