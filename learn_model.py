"""
LearningModel is class that is used to make a model that
Continually builds and expands on neural nets in an attempt to build
a neural net to interpret the movement of the solution space from the
data provided

Methods Employed:
The neural networks are basic feedforward artificial neural networks
The neural network manipulation algorithm is based off of a paper
written by Kenneth O. Stanley and Risto Miikkulainen by the name
of NEAT (NeuroEvolution of Augmenting Topologies). This system is
interesting to us as it is possible to 'optimize and complexify solutions
simultaneously.' Inspiration for this comes from SethBling and his
MarI/O.

Documentation Points:
1.  The file format for saving and restoring generations is described below
    The format is JSON inspired with implicit attributes based on the structure of
    the developed classes. This sacrifices readability in order to decrease
    file size by a factor of at least 2:1 (less conservative estimates suggest
    closer to a factor 4:1).

    Files will begin with '{' and end with '}' and will encompass exactly one pool
    A default pool will look as follows '{[],0,0,0,0,0}'

    A pool with one species with one genome with one default gene
    will look as follows
    '{[{0,0,[{[{0,0,0,F,0}],0,0,0,0,0,0,0;0;0;0;0;0;0}],0}],0,0,0,0,0}'

    One important note is that the network attribute of the Genome class will
    always be absent from this file. Storing the network class would be
    redundant and we currently value efficient memory usage over redundancy.
    The purpose of this file structure is decreased readability, in order to
    conserve the limited memory we have available to us.
    Files of this type will be marked .ion for Implicit Object Notation.
2.  After thorough testing it has been confirmed that the line
    'actualized_pool = self.actualize_objects(file_contents, labels, file_structure)'
    in the function restore_generation does in fact return values, despite what the
    warning says. This warning seems to stem from the fact that the initial values
    of finished_objects are set to 'None', because when these values are changed to
    0 the warning disappears. More information needs to be collected on this, as of
    now it is being marked as a minor bug.
3.  Two adjacency lists were used to represent the graph. For one adjacency list
    at index i there were a list of elements such that any element j in that list
    represents an edge (i,j) in the graph. In the second adjacency list at index i
    there was a list of elements such that any element j in that list represents an
     edge (j,i) in the graph. In comparison to a single adjacency list, this has the
     advantage of improving the average case run time. For a single adjacency list
     the best, average, and worse case search time for either incoming or outgoing
     nodes (depending on which direction on the adjacency lists' implementation)
     is O(E). For a double adjacency list the best case search time is O(1) and the
     average case search time depends on the sparsity of the directed acyclic graph,
     but can be no worse O(E). Having twice the adjacency lists comes with the obvious
     side-effect of double the space complexity. Using two adjacency lists has
     the advantage of saving space, as an adjacency matrix costs O(n^2) whereas an
     adjacency list costs O(n). On top of that searching for all the edges off of
     a node takes best, average, and worst case O(n), whereas the double adjacency
     list does better than that (as discussed before)
4.  I tried to find literature about the most meaningful way to select individuals
    to engage in crossover given speciation information, but was unable to find anything.
    After consideration I have chosen to allow crossover between any two individuals
    in the population. The reason is this, forcing crossover between individuals of the
    same species would mean that we would more efficiently find the local optimum that the
    species is converging too, but we will likely miss out on any optimum that are mixtures
    of functionality across species. Specifically, forcing crossover intra-species
    means that we converge to local optimum faster, but since the so called "Pareto Front"
    we maintain is actually a subset of the Pareto Front we are more likely to miss out on
    the global optimum.

Ideas Used:
1.  NEAT (obvious)
2.  Kahn's Topological Sort (Topological ordering of genes)
3.  Fitness Proportionate Selection
4.  Implicit Object Notation (File storage)
5.  Niching or Speciation which is inherent in NEAT
6.  Generative Adversarial Networks
7.  Layers of Abstraction (Truthfully, this component was not
    needed, but I wanted practice)(This is seen in using
    GameModel's to hide the underlying neural networks)


author: Andrew Butler
"""

# TODO REVISE DOCSTRING TO INCLUDE PROOF OF WORST CASE DENSITY IN A DAG
# TODO MOVE MAJORITY OF HEADER DOCSTRING TO README FILE
# TODO CREATE NEW ERRORS WHICH CAN BE REPLACED FOR THE NotImplemented
# TODO ADD DROPOUT NEURONS (FROM DATASKEPTIC PODCAST)
# TODO LIST OF MUTATIONS IN CURRENT GENERATIONS TO ENSURE A MUTATION
# DOES NOT APPEAR MORE THAN ONCE
# TODO FOR FUTURE: REMOVE STALE SPECIES AND REPLACE WITH BASIC GENOMES
# TO FACILITATE RANDOM RESTARTING WHEN THE POPULATION STALES OUT
import math
import random
import copy


class Pool:
    """
    The complete generational pool of species and data about it

    species:        The list of species in this pool
    generation:     The number of iterations of generations that
                    this pool has already undergone
    innovation:     The next unused value to be given to an innovation
    current_species:The current species being evaluated
    current_genome: The current genome being evaluated
    max_fitness:    The maximum fitness for any genome in the pool
    """

    def __init__(self, species=None, generation=0, innovation=0, current_species=0,
                 current_genome=0, max_fitness=0):
        if species is None:
            species = []
        self.species = species
        self.generation = generation
        self.innovation = innovation
        self.current_species = current_species
        self.current_genome = current_genome
        self.max_fitness = max_fitness

    def to_string(self):
        """
        Returns string representation of the Genome in the
        format that the .ion file requires. Specified in
        documentation point 1 at the heading docstring
        (to be moved to README file)

        :return:    String representation of the Genome
        """

        to_return = "{["
        if len(self.species) > 0:
            to_return += self.species[0].to_string()
        for species_iter in range(1, len(self.species)):
            to_return += "," + self.species[species_iter].to_string()
        to_return += "]," + str(self.generation) + "," + str(self.innovation)
        to_return += "," + str(self.current_species) + "," + str(self.current_genome)
        to_return += "," + str(self.max_fitness) + "}"

        return to_return

    def get_innovation(self):
        """
        Increments and returns the innovation number

        :return:    The current innovation number
        """

        to_return = self.innovation
        self.innovation += 1
        return to_return


class Species:
    """
    A complete set of genomes that evolved together as a species
    and data about it

    top_fitness:    The fitness of the genome with the highest fitness in
                    this species.
    staleness:      The number of generations this genome has gone without
                    improving its best fitness
    genomes:        List of genomes in this species
    average_fitness:The average fitness of genomes in this species
    """

    def __init__(self, top_fitness=0, staleness=0, genomes=None, average_fitness=0.0):
        if genomes is None:
            genomes = []
        self.top_fitness = top_fitness
        self.staleness = staleness
        self.genomes = genomes
        self.average_fitness = average_fitness

    def to_string(self):
        """
        Returns string representation of the Genome in the
        format that the .ion file requires. Specified in
        documentation point 1 at the heading docstring
        (to be moved to README file)

        :return:    String representation of the Genome
        """

        to_return = "{" + str(self.top_fitness) + "," + str(self.staleness) + ",["
        if len(self.genomes) > 0:
            to_return += self.genomes[0].to_string()
        for genome_iter in range(1, len(self.genomes)):
            to_return += "," + self.genomes[genome_iter].to_string()
        to_return += "]," + str(self.average_fitness) + "}"

        return to_return


class Genome:
    """
    A set of genes that together form a genome and data about it

    genes:          The list of genes that represent the genome
    network:        The actual Network representation of the genome
                    if it has been created
    mutation_rates: The MutationRate associated with this genome
    fitness:        The current fitness of the genome
    shared_fitness: The fitness when considering the species of the genome
    num_inputs:     The number of input neurons in the genome's network
    num_outputs:    The number of output neurons in the genome's network
    max_neuron:     The number of neurons in the genome's network
    max_innovation: The largest innovation number of a gene in the genome
    global_rank:    The current rank of the genome
    topological_order:A list that represents the topological order
                    of the neurons
    """

    def __init__(self, genes=None, fitness=0, shared_fitness=0, network=None,
                 num_inputs=0, num_outputs=0, max_neruon=0, global_rank=0,
                 max_innovation=0, mutation_rates=None, topological_order=None):
        if genes is None:
            genes = []
        if network is None:
            network = []
        if mutation_rates is None:
            mutation_rates = MutationRates()
        if topological_order is None:
            topological_order = []
        self.genes = genes
        self.fitness = fitness
        self.shared_fitness = shared_fitness
        self.network = network
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.max_neuron = max_neruon
        self.global_rank = global_rank
        self.max_innovation = max_innovation
        self.mutation_rates = mutation_rates
        self.topological_order = topological_order

    def add_gene(self, gene):
        """
        Adds a gene to the genome and changes the necessary
        parameters associated

        :param gene:    The gene to be added
        :return:
        """

        lower_bound = -1
        upper_bound = -1

        for order_iter in range(0, len(self.topological_order)):
            cur_node = self.topological_order[order_iter]
            if cur_node == gene.out:
                lower_bound = order_iter
            if cur_node == gene.into:
                upper_bound = order_iter

        if (lower_bound == -1) | (upper_bound == -1):
            raise NotImplementedError("Adding genes for new neural networks is not"
                                      "currently supported")
        else:

            if (gene.into == self.max_neuron) | (gene.out == self.max_neuron):
                self.max_neuron += 1
            self.genes.append(gene)

            random_num = random.randint(lower_bound + 1, upper_bound)
            new_order = self.topological_order[0:random_num]
            new_order.append(gene)
            new_order.extend(self.topological_order[random_num:])
            self.topological_order = new_order

            if gene.innovation > self.max_innovation:
                self.max_innovation = gene.innovation

    def to_string(self):
        """
        Returns string representation of the Genome in the
        format that the .ion file requires. Specified in
        documentation point 1 at the heading docstring
        (to be moved to README file)

        :return:    String representation of the Genome
        """

        to_return = "{["
        if len(self.genes) > 0:
            to_return += self.genes[0].to_string()
        for gene_iter in range(1, len(self.genes)):
            to_return += "," + self.genes[gene_iter].to_string()
        to_return += "]," + str(self.fitness) + "," + str(self.shared_fitness)
        to_return += "," + str(self.num_inputs) + "," + str(self.num_outputs)
        to_return += "," + str(self.max_neuron) + "," + str(self.global_rank)
        to_return += "," + self.mutation_rates.to_string() + "}"

        return to_return


class Gene:
    """
    A description of a connection between to neurons and data about it

    into:       The neuron to which the connection goes
    out:        The neuron from which the connection starts
    weight:     The level of distortion with which the value from
                'out' will receive when being considered for 'into'
    enabled:    Whether or not this gene is currently a part of the network
    average_fitness:    The average fitness associated with this specific gene
    innovation: The innovation number unique to this gene across all genomes
    """

    def __init__(self, into=0, out=0, weight=0.0, enabled=False, innovation=0):
        self.into = into
        self.out = out
        self.weight = weight
        self.enabled = enabled
        self.innovation = innovation

    def to_string(self):
        """
        Returns string representation of the Gene in the
        format that the .ion file requires. Specified in
        documentation point 1 at the heading docstring
        (to be moved to README file)


        :return:    String representation of the Gene
        """

        to_return = "{" + str(self.into) + "," + str(self.out) + "," + str(self.weight) + ","
        if self.enabled:
            to_return += "T"
        else:
            to_return += "F"
        to_return += "," + str(self.innovation) + "}"

        return to_return


class Network:
    """
    A set of neurons that make up a network.

    num_inputs:     The number of neurons that are in the input layer
    num_outputs:    The number of neurons that are in the output layer
    neurons:        List of neurons in the neural network. The list is
                    ordered from first to last topological order except
                    for the output layer which immediately follows the
                    input layer
    neurons_sorted: List of neurons in the neural network. The list is
                    ordered from the neuron labeled 1 to the neuron
                    labeled the length of the network (including
                    disabled nodes)
    """

    def __init__(self, num_inputs=0, num_outputs=0, neurons=None, neurons_sorted=None):
        if neurons is None:
            neurons = []
        if neurons_sorted is None:
            neurons_sorted = []
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.neurons = neurons
        self.neurons_sorted = neurons_sorted


class Neuron:
    """
    A node in a neural network which models the functions of a dendrite,
    by having the nodes through which the neural cell receives data.

    incoming_neurons:   List of the labels of the neural nodes that provide
                        information for this neuron
    weights:            List of the weights of the connections from other
                        neurons. The weight at index i corresponds to the
                        connection from the neuron at incoming_neurons index
                        i to the current neuron
    value:              The associated value in the neural network based on
                        the latest use of inputs
    """

    def __init__(self, incoming_neurons=None, weights=None, value=0.0):
        if incoming_neurons is None:
            incoming_neurons = []
        if weights is None:
            weights = []
        self.incoming_neurons = incoming_neurons
        self.weights = weights
        self.value = value


class MutationRates:
    """
    A set of values denoting the rates of specific mutations

    connection:     The likelihood of a mutation where a connection is added
                    between two pre-existing neurons
    link:           The likelihood of a mutation where a connections weight
                    is either perturbed or randomly chosen
    bias:           The likelihood of a mutation where a connection from the
                    bias node to a non-input node is made
    node:           The likelihood of a mutation where a connection pair
                    (out, into) is disabled, a node 'new' is created, and
                    two connection pairs are created (out, new) and (new, into)
    enable:         The likelihood of a mutation where a certain gene is
                    disabled.
    disbale:        The likelihood of a mutation where a certain gene is
                    enabled.
    step:           The maximum change in either direction for the weight of
                    a gene if it is being perturbed

    """

    def __init__(self, connection=0.0, link=0.8, bias=0.0, node=0.0, enable=0.0,
                 disable=0.0, step=0.0):
        self.connection = connection
        self.link = link
        self.bias = bias
        self.node = node
        self.enable = enable
        self.disable = disable
        self.step = step

    def to_string(self):
        """
        Returns string representation of the Mutation Rate in the
        format that the .ion file requires. Specified in
        documentation point 1 at the heading docstring
        (to be moved to README file)


        :return:    String representation of the Mutation Rate
        """
        to_return = str(self.connection) + ";" + str(self.link) + ";" + str(self.bias)
        to_return += ";" + str(self.node) + ";" + str(self.enable) + ";"
        to_return += str(self.disable) + ";" + str(self.step)

        return to_return


class ModelConstants:
    """
    Constants that describe the model

    inputs:         The number of input nodes
    outputs:        The number of output nodes
    population_size:The size of the population during a generation
    parallel_evals: The number of networks being evaluated at the same time
    games_per_genome:The number of games each genome plays during a generation
    """
    def __init__(self, inputs, outputs, population_size, parallel_evals, games_per_gen):
        self.inputs = inputs
        self.outputs = outputs
        self.population_size = population_size
        self.parallel_evals = parallel_evals
        self.games_per_genome = games_per_gen

class GameConstants:
    """
    Constants that describe how the games should be played

    parallel_evals: The number of networks being evaluated at the same time
    games_per_genome:The number of games each genome plays during a generation
    evals_per_game: The number of evaluations on a network each game
    """
    def __init__(self, parallel_evals, games_per_gen, hands_per_game):
        self.parallel_evals = parallel_evals
        self.games_per_genome = games_per_gen
        self.hands_per_game = hands_per_game


class SpeciationValues:
    """
    Coefficients for use in calculating the similarity of genomes

    disjoint_coeff:     The coefficient for the penalty on disjoint genes
    excess_coeff:       The coefficient for the penalty on excess genes
    weight_coeff:       The coefficient for the penalty on the difference
                        in weights between two genes of the same innovation
    compat_constant:    The constant for the allowed compatibility difference
                        between genomes of the same species
    """
    def __init__(self, disjoint_coeff, excess_coeff, weight_coeff, compat_constant):
        self.disjoint_coeff = disjoint_coeff
        self.excess_coeff = excess_coeff
        self.weight_coeff = weight_coeff
        self.compat_constant = compat_constant


class ModelRates:
    """
    A collection of rates on how the learning model should change the
    genomes it contains

    perturb_rate:   The chance that mutated links will perturb
                    (This value is usually around 0.9)
    """
    def __init__(self, perturb_rate, interspecies_mating_rate):
        self.perturb_rate = perturb_rate
        self.interspecies_mating_rate = interspecies_mating_rate


class LearningModel:
    def __init__(self, model_constants, game_model,
                 mutation_rates, speciation_values, model_rates):
        """
        Saves the initial parameters

        :param model_constants: The constants associated with the neural network model
        :param game_model:  The method that evaluates a neural networks fitness
        :param mutation_rates:  The rates that the neural networks originally get
                                mutated at
        :param speciation_values:   The coefficients for the similarity function
                                as well as the compatibility constant
        :param model_rates:     The rates of change that the learning model forces
                                on the genomes it maintains
        """

        if mutation_rates is None:
            mutation_rates = MutationRates()
        self.model_constants = model_constants
        self.game_model = game_model
        self.mutation_rates = mutation_rates
        self.speciation_values = speciation_values
        self.model_rates = model_rates
        self.pool = Pool()
        self.population = []

    def set_pool(self, value):
        self.pool = value

    def set_mutation_rates(self, value):
        self.mutation_rates = value

    def get_innovation(self):
        """
        Increments and returns the innovation number

        :return:    The current innovation number
        """

        return self.pool.get_innovation()

    @staticmethod
    def sigmoid(value):
        """
        Returns the sigmoid of 'value'
        :param value:   Value for sigmoid function to
        :return:        Return sigmoid of value
        """
        denom = 1 + math.exp(-4.9*value)
        return (2/denom)-1

    def save_generation(self, file_name):
        """
        Stores the current generation into a file
        Information about how the data is represented
        is present in documentation point 1 at the
        heading docstring (to be moved to README file)

        :param file_name:   Name of file to be saved to
        :return:            The string representation of
                            the pool
        """

        to_file_write = self.pool.to_string()
        file_write = open(file_name, "w+")
        file_write.write(to_file_write)
        return to_file_write

    # TODO Figure out actualized_pool warning
    def restore_generation(self, file_name):
        """
        Restores a generation from a file. This file must follow the
        .ion file structure.

        :param file_name:   The path of the file to be restored
        """
        read_file = open(file_name)
        file_contents = read_file.read()
        file_structure = sorted(self.find_file_structure(file_contents))
        labels = self.label_structure(file_structure)

        actualized_pool = self.actualize_objects(file_contents, labels, file_structure)
        self.set_pool(actualized_pool)

    def find_file_structure(self, file_contents):
        """
        Takes in a string from an archived file and returns the start and
        finish of every object represented in this file

        :param file_contents:   A string representing the file structure
                                of Pools, Species, Genomes, and Genes
        :return:                Returns a tuple representing the start
                                and finish of the representation of
                                every object in the file given
        """

        cur_letter = 0
        end_file = len(file_contents)
        list_of_components = []
        start_of_layers = []    # Stack used to maintain 'start' structure markers

        # Iterates through the string in order looking for structure markers
        while not cur_letter == end_file:
            letter = file_contents[cur_letter: cur_letter + 1]
            if (letter == "{") | (letter == "["):
                start_of_layers.append(cur_letter)
            elif (letter == "}") | (letter == "]"):
                list_of_components.append((start_of_layers.pop(), cur_letter))
            cur_letter += 1

        return list_of_components

    def label_structure(self, file_struct):
        """
        Returns an array where an element at index 'i' represents the structural
        depth of the object at file_struct[i]. The structural depth of an object
        is how deeply embedded within other lists and objects it is. The object
        Cat in {{{Cat},Dog}} would have a structural depth of 2 for example. The
        structural depth is important as it identifies the class of that object.

        :param file_struct: A list of (start, end) pairs where each (start, end)
                            pair is the index of the start and end of an object
        :return:            A list of labels indicating the structural depth of
                            the objects represented in file_struct
        """

        list_iter = 0
        num_structs = len(file_struct)
        open_structs = []          # Stack for maintaining structural depth
        labels = [-1 for _ in range(0, len(file_struct))]

        # Iterates through the file structure to find the structural depth of each object
        while list_iter < num_structs:
            # If open_structs is empty, start the stack
            if not open_structs:
                open_structs.append(list_iter)
            else:
                last_struct = file_struct[open_structs[len(open_structs) - 1]]

                # If the last structure seen has an end that is after the current structures'
                # beginning, add the current structure to the stack
                if last_struct[1] > file_struct[list_iter][0]:
                    open_structs.append(list_iter)
                # Else the last structure ends before the current one starts, so remove the last
                # structure from the stack and save its structural depth
                else:
                    popped = open_structs.pop()
                    labels[popped] = len(open_structs)
                    list_iter -= 1

            list_iter += 1

        while not open_structs == []:
            cur_item = open_structs.pop()
            labels[cur_item] = len(open_structs)

        return labels

    def actualize_objects(self, file_contents, labels, file_struct):
        """
        Restores the objects that are contained in file_contents
        with the given meta-data 'labels' and 'file_struct' into
        a Pool. Returns that Pool

        :param file_contents:   The string representation of the file
                                being restored
        :param labels:          The set of labels that denote the structural
                                depth of the 'i'th object
        :param file_struct:     The set of (start, end) pairs that denote the
                                start and end of the 'i'th object, sorted in
                                increasing order of the 'start'
        :return:                The Pool that is stored in the file
        """

        if len(labels) == 0:
            return None

        cur_index = 0
        last_index = len(labels)
        finished_objects = [None for _ in range(0, last_index)]

        # Iterate through every object in an attempt to label them
        while not cur_index >= last_index:
            cur_depth = labels[cur_index]
            # Protection against when we get to the last element
            if cur_index == last_index - 1:
                next_depth = -1
            else:
                next_depth = labels[cur_index + 1]

            # If the current items depth is greater than the next items depth then
            # we have reached the leaf of a branch in the file structure.
            if cur_depth > next_depth:
                backtrack_index = cur_index

                # Iterate backwards through the objects until we reach the lowest
                # common ancestor with another branch in the file structure
                while (backtrack_index >= 0) & (labels[backtrack_index] > next_depth):
                    # We take advantage of the objects structure, by noting that every other
                    # object must be a list. Even numbers are Pools, Species, Genomes, and Genes.
                    # While odd numbers are lists.
                    # If the structural depth is even, then restore the object
                    if labels[backtrack_index] % 2 == 0:
                        obj_contents = []
                        obj_content_bounds = None

                        if backtrack_index + 1 < len(labels):
                            obj_contents = finished_objects[backtrack_index + 1]
                            obj_content_bounds = file_struct[backtrack_index + 1]

                        obj = self.restore_object(file_contents, labels[backtrack_index],
                                                  file_struct[backtrack_index],
                                                  obj_contents,
                                                  obj_content_bounds)
                        finished_objects[backtrack_index] = obj
                    # Else if the structural depth is odd, then compile all the elements
                    # of the list
                    else:
                        # If the list is the last object then we know it is empty
                        if backtrack_index >= len(labels) - 1:
                            finished_objects[backtrack_index] = []
                        # If the next object is not exactly one structural depth lower
                        # then the list is empty
                        elif not labels[backtrack_index] == labels[backtrack_index + 1] - 1:
                            finished_objects[backtrack_index] = []
                        # Else add objects all object oof exactly one structural depth lower
                        # until you reach an object that is one structural depth higher
                        else:
                            list_depth = labels[backtrack_index]
                            list_index = backtrack_index + 1
                            compiled_list = []

                            if list_index < len(labels):
                                back_depth = labels[list_index]

                                while list_depth < back_depth:
                                    if list_depth == labels[list_index] - 1:
                                        compiled_list.append(finished_objects[list_index])
                                    list_index += 1
                                    if list_index < len(labels):
                                        back_depth = labels[list_index]
                                    else:
                                        back_depth = list_depth

                            finished_objects[backtrack_index] = compiled_list
                    backtrack_index -= 1
            cur_index += 1
        return finished_objects[0]

    def restore_object(self, file_contents, struct_depth, struct_bounds,
                       struct_contents, content_bounds):
        """
        This object takes in multiple items of meta-data about file_contents
        and uses this to efficiently restore the object to memory

        :param file_contents:       The string representation of the object
                                    being restored
        :param struct_depth:        The structural depth of the object in the
                                    file being restored. This corresponds to
                                    the class of the object being restored
        :param struct_bounds:       A (start, end) pair where the start denotes
                                    the leading '{' and the end denotes the
                                    trailing '}' of the object
        :param struct_contents:     A list of objects that have been restored
                                    that are owned by the object currently
                                    being restored
        :param content_bounds:      A (start, end) pair where the start denotes
                                    the leading '[' and the end denotes the
                                    trailing ']' of the object's content
        :return:                    Returns the object being restored
        """

        file_len = len(file_contents)
        # If the object to be restored has a structural depth of 0, then it is a Pool
        if struct_depth == 0:
            cur_pool = Pool()
            cur_pool.species = struct_contents
            cur_item_index = content_bounds[1] + 2
            end_item_index = cur_item_index + file_contents[cur_item_index:file_len].find("}")

            list_items = file_contents[cur_item_index:end_item_index].split(",")

            cur_pool.generation = int(list_items.pop(0))
            cur_pool.innovation = int(list_items.pop(0))
            cur_pool.current_species = int(list_items.pop(0))
            cur_pool.current_genome = int(list_items.pop(0))
            cur_pool.max_fitness = int(list_items.pop(0))
            return cur_pool

        # If the object to be restored has a structural depth of 2, then it is a Species
        elif struct_depth == 2:
            cur_species = Species()

            cur_item_index = struct_bounds[0] + 1
            next_item_index = cur_item_index + file_contents[cur_item_index: file_len].find(",")
            cur_species.top_fitness = int(file_contents[cur_item_index: next_item_index])

            cur_item_index = next_item_index + 1
            next_item_index = cur_item_index + file_contents[cur_item_index: file_len].find(",")
            cur_species.staleness = int(file_contents[cur_item_index: next_item_index])

            cur_species.genomes = struct_contents

            cur_item_index = content_bounds[1] + 2
            next_item_index = cur_item_index + file_contents[cur_item_index: file_len].find("}")
            cur_species.average_fitness = float(file_contents[cur_item_index: next_item_index])
            return cur_species

        # If the object to be restored has a structural depth of 4, then it is a Genome
        elif struct_depth == 4:
            cur_genome = Genome()

            cur_genome.genes = struct_contents
            cur_genome.topological_order = self.topological_sort(struct_contents)

            max_innov = 0
            # Finds the max innovation number from the genes in the genome
            for gene in cur_genome.genes:
                if gene.innovation > max_innov:
                    max_innov = gene.innovation

            cur_genome.max_innovation = max_innov

            cur_item_index = content_bounds[1] + 2
            end_item_index = cur_item_index + file_contents[cur_item_index: file_len].find(";")

            list_items = file_contents[cur_item_index: end_item_index].split(",")

            cur_genome.fitness = int(list_items.pop(0))
            cur_genome.shared_fitness = int(list_items.pop(0))
            cur_genome.num_inputs = int(list_items.pop(0))
            cur_genome.num_outputs = int(list_items.pop(0))
            cur_genome.max_neuron = int(list_items.pop(0))
            cur_genome.global_rank = int(list_items.pop(0))

            cur_item_index = end_item_index + 1
            end_item_index = cur_item_index + file_contents[cur_item_index: file_len].find("}")

            hold_over = list_items.pop(0)
            list_items = file_contents[cur_item_index: end_item_index].split(";")

            cur_mutation_rate = MutationRates()
            cur_mutation_rate.connection = float(hold_over)
            cur_mutation_rate.link = float(list_items.pop(0))
            cur_mutation_rate.bias = float(list_items.pop(0))
            cur_mutation_rate.node = float(list_items.pop(0))
            cur_mutation_rate.enable = float(list_items.pop(0))
            cur_mutation_rate.disable = float(list_items.pop(0))
            cur_mutation_rate.step = float(list_items.pop(0))

            cur_genome.mutation_rates = cur_mutation_rate

            return cur_genome

        # If the object to be restored has a structural depth of 6, then it is a Gene
        elif struct_depth == 6:

            cur_gene = Gene()

            cur_item_index = struct_bounds[0] + 1
            end_item_index = cur_item_index + file_contents[cur_item_index:file_len].find("}")

            list_items = file_contents[cur_item_index:end_item_index].split(",")

            cur_gene.into = int(list_items.pop(0))
            cur_gene.out = int(list_items.pop(0))
            cur_gene.weight = float(list_items.pop(0))
            cur_gene.enabled = list_items.pop(0) == "T"
            cur_gene.innovation = int(list_items.pop(0))

            return cur_gene

        # Else the Object has been mislabeled, raise an error
        else:
            raise NotImplementedError("The file given is not formatted correctly")

    def topological_sort(self, genome):
        """
        A topological sort (using Kahn's Algorithm) to create an order of
        the poset that will be used to ensure the directed acyclic graph
        stays acyclic when adding edges. The design decisions for this
        algorithm are explained in documentation point 3.

        :param genome:   A list of genes that represent the genome
        :return:        A list representing the topological order
                        of the nodes in the neural network
        """

        copy_genes = list.copy(genome.genes)

        adj_list = [[] for _ in range(0, genome.max_neuron)]

        # This adjacency list is atypical in that at each index i
        # there is a list of indices such that each element j
        # in that list denotes an edge (j,i) in the graph.
        # (As opposed to the typical (i,j) representation)
        adj_list_rev = [[] for _ in range(0, genome.max_neuron)]

        sorted_list = []
        cur_list = []
        next_list = []

        # Fills the two adjacency lists with the edges from the genes
        for gene in copy_genes:
            adj_list_rev[gene.into].append(gene.out)
            adj_list[gene.out].append(gene.into)

        # Creates the initial list from the nodes with no incoming edges
        # There is required to be at least one or the neural network is invalid
        for node_num in range(0, genome.max_neuron):
            if not adj_list_rev[node_num]:
                cur_list.append(node_num)

        # While the list is not empty, continue to pick elements at random and
        # add them to the topologically ordered list. Remove the edges coming off
        # of the chosen element and add any nodes that no longer have incoming edges
        # to the next list to be evaluated
        while cur_list:
            random_num = random.randint(0, len(cur_list) - 1)
            cur_node = cur_list.pop(random_num)
            sorted_list.append(cur_node)

            # Removing edges and checking for existence of incoming edges
            while adj_list[cur_node]:
                cur_edge = adj_list[cur_node].pop(0)
                adj_list_rev[cur_edge].remove(cur_node)
                if not adj_list_rev[cur_edge]:
                    next_list.append(cur_edge)

            # If finished with the current list, move onto the next list
            if not cur_list:
                cur_list = next_list
                next_list = []

        return sorted_list

    def copy_genome(self, genome):
        """
        Deep copy of 'genome'
        :param genome:  Genome to be copied
        :return:        Returns a deep copy of the genome
        """

        genome_copy = Genome()
        for gene in genome.genes:
            genome_copy.genes.append(self.copy_gene(gene))
        genome_copy.shared_fitness = genome.shared_fitness
        genome_copy.fitness = genome.fitness
        genome_copy.global_rank = genome.global_rank
        genome_copy.max_neuron = genome.max_neuron
        genome_copy.mutation_rates = self.copy_mutation_rates(genome.mutation_rates)
        genome_copy.network = self.copy_network(genome.network)
        genome_copy.topological_order = copy.deepcopy(genome.topological_order)
        return genome_copy

    def copy_gene(self, gene):
        """
        Deep copy of 'gene'
        :param gene:    Gene to be copied
        :return:        Returns a deep copy of the gene
        """

        copy_gene = Gene()
        copy_gene.into = gene.into
        copy_gene.out = gene.out
        copy_gene.weight = gene.weight
        copy_gene.enabled = gene.enabled
        copy_gene.innovation = gene.innovation
        return copy_gene

    def copy_mutation_rates(self, mut_rates):
        """
        Deep copy of 'mut_rates'
        :param mut_rates:   Rates of Mutation to be copied
        :return:            Returns a deep copy of the mutation rates
        """

        copy = MutationRates()
        copy.connection = mut_rates.connection
        copy.link = mut_rates.link
        copy.bias = mut_rates.bias
        copy.node = mut_rates.node
        copy.enable = mut_rates.enable
        copy.disable = mut_rates.disable
        copy.step = mut_rates.step
        return copy

    def copy_network(self, network):
        """
        Deep copy of 'network'
        :param network:     Network to be copied
        :return:            Returns a deep copy of the mutation rates
        """

        network_copy = Network()
        for neuron in network.neurons:
            network_copy.neurons.append(self.copy_neuron(neuron))
        for neuron2 in network.neurons_sorted:
            network_copy.neurons_sorted.append(self.copy_neuron(neuron2))
        network_copy.num_inputs = network.num_inputs
        network_copy.num_outputs = network.num_outputs
        return network_copy

    def copy_neuron(self, neuron):

        neuron_copy = Neuron()
        neuron_copy.incoming_neurons = copy.deepcopy(neuron.incoming_neurons)
        neuron_copy.weights = copy.deepcopy(neuron.weights)
        neuron_copy.value = neuron.value

        return neuron_copy

    def create_network(self, genome):
        """
        Creates and returns the network that is
        modeled by 'genome'
        :param genome:      Any instantiation of Genome
        :return:            A network modeled by 'genome'
        """
        network = Network()
        network.num_inputs = genome.num_inputs
        network.num_outputs = genome.num_outputs
        num_nodes = genome.max_neuron
        neuron_order = genome.topological_order

        neuron_list = []

        for x in range(0, num_nodes):
            neuron_list.append(Neuron())

        for gene in genome.genes:
            if gene.enabled:
                cur_neuron = neuron_list[gene.out]
                cur_neuron.incoming_neurons.append(gene.into)
                cur_neuron.weights.append(gene.weight)

        for neuron_iter in range(0, genome.num_inputs+genome.num_outputs):
            network.neurons.append(neuron_list[neuron_iter])
            neuron_order.remove(neuron_iter)

        for node in neuron_order:
            network.neurons.append(neuron_list[node])

        network.neurons_sorted = neuron_list

        return network

    def evaluate_neuron_values(self, network, inputs):
        """
        Assigns a value to each neuron based on the structure
        of the network and the inputs that are given

        :param network:     The network to be evaluated
        :param inputs:      The set of numbers that are
                            the values of the input neurons
        """

        # Assigns the input neurons their values
        for input_iter in range(0, network.num_inputs):
            network.neurons[input_iter] = inputs[input_iter]

        hidden_start = network.num_inputs + network.num_outputs

        # Finds out the value of every next neuron based on previous neurons'
        # values and the structure of the graph
        for hidden_iter in range(hidden_start, len(network.neurons)):
            cur_node = network.neurons[hidden_iter]
            total_value = 0

            for incoming_iter in range(0, len(cur_node.incoming_neurons)):
                cur_connection = cur_node.incoming_neurons[incoming_iter]
                cur_value = network.neurons_sorted[cur_connection].value
                total_value += cur_value * cur_node.weights[incoming_iter]

            cur_node.value = self.sigmoid(total_value)

    def basic_genome(self, inputs, outputs):
        """
        Creates and returns a default genome with no connections

        :param inputs:  The number of input variables for the neural network
        :param outputs: The number of output variables for the neural network
        :return:        A default genome with no connections
        """
        genome = Genome(None, 0, 0, None, inputs + 1, outputs, inputs + outputs + 1,
                        0, self.copy_mutation_rates(self.mutation_rates), None)

        return genome

    def basic_genome_connected(self, inputs, outputs):
        """
        Creates and returns a default genome with connections from
        every input node to every output node

        :param inputs:  The number of input variables for the neural network
        :param outputs: The number of output variables for the neural network
        :return:        A default genome with connections from every input node
                        to every output node
        """

        genome = self.basic_genome(inputs, outputs)

        # Adds genes from every input node to every output node
        for input_iter in range(0, inputs):
            for output_iter in range(0, outputs):
                genome.add_gene(Gene(output_iter + inputs + 1, input_iter,
                                     (random.random() * 4) - 2, True, 0))

        self.mutate(genome)

        return genome

    def mutate(self, genome):
        """
        Runs through all the possible mutation types on
        the genome

        :param genome:  The genome to be mutated
        """

        methods = [self.mutate_enable_disable, self.mutate_weights,
                   self.mutate_connection, self. mutate_node, self.mutate_bias]

        for mutation in methods:
            mutation(genome)

    def mutate_connection(self, genome):
        """
        Adds a connection (gene) between two nodes with a random weight
        between -2 and 2. The number of connections added is based on
        the connection mutation rate of the genome

        :param genome:      The genome to which the gene is being added
        """

        mutation_rate = genome.mutation_rates.connection

        while mutation_rate > 0:
            if random.random() < mutation_rate:

                random_list = [x for x in range(0, len(genome.max_neuron)-1)]
                random_node1 = random.randint(0,len(random_list))
                random_list.remove(random_node1)
                random_node2 = random.randint(0,len(random_list))

                # Ensures there are no recurrent connections
                if (genome.topological_order.index(random_node1) >
                        genome.topological_order.index(random_node2)):
                    temp = random_node1
                    random_node1 = random_node2
                    random_node2 = temp

                random_num = (random.random()*4)-2
                new_gene = Gene(random_node2, random_node1, random_num, True, self.get_innovation)
                genome.genes.append(new_gene)

    def mutate_node(self, genome):
        """
        Disables a gene and adds two genes in its place. This creates an
        intermediary node where there was none before adding to the
        complexity of the neural network. The number of nodes added
        is based on the genome's mutation rate for nodes

        :param genome:      The genome that is to be mutated
        """

        mutation_rate = genome.mutation_rates.node

        while mutation_rate > 0:
            if random.random() < mutation_rate:
                old_gene = genome.genes[random.randint(0, len(genome.genes)) - 1]

                # Ensures that the gene chosen is enabled and not connected to a bias node
                if old_gene.enabled & (not old_gene.out == genome.num_inputs):

                    # The connection into the new node has a weight of 1 while the connection into
                    # the old_gene's into node receives the old_gene's weight.
                    # This minimizes the immediate effect on the networks fitness
                    gene_split1 = Gene(genome.max_neuron, old_gene.out, 1.0, True, 0)
                    gene_split2 = Gene(old_gene.into, genome.max_neuron, old_gene.weight, True, 0)

                    old_gene.enabled = False

                    genome.genes.append(gene_split1)
                    genome.genes.append(gene_split2)

    def mutate_weights(self, genome):
        """
        Mutate the weights of the connections between the nodes.
        This is done either through randomly selecting a new weight
        or perturbing the current weight based on the mutation.

        :param genome:      The genome to be mutated
        """
        step = genome.mutation_rates.step
        perturb_rate = self.model_rates.perturb_rate

        for gene in genome.genes:
            if random.random() < perturb_rate:
                gene.weight = gene.weight + (2 * step * random.random()) - step
            else:
                gene.weight = (random.random() * 4) - 2

    def mutate_enable_disable(self, genome):
        """
        Runs the enable and disable mutation on the
        genome, based on the mutation rates

        :param genome:  The genome to be mutated
        """

        enable_rate = genome.mutation_rates.enable
        disable_rate = genome.mutation_rates.disable

        while enable_rate > 0:
            if random.random() < enable_rate:
                self.mutate_enable_disable_choice(genome, True)
            enable_rate -= 1

        while disable_rate > 0:
            if random.random() < disable_rate:
                self.mutate_enable_disable_choice(genome, False)
            disable_rate -= 1

    def mutate_enable_disable_choice(self, genome, is_enable):
        """
        Change a single gene that matches is_enable to
        not is_enable. This gene is chosen randomly


        :param genome:      The genome to be mutated
        :param is_enable:   If we want to mutate an enabled node
        """

        choices = []

        for gene in genome.genes:
            if gene.enabled == is_enable:
                choices.append(gene)

        random_num = random.randint(0, len(choices) - 1)
        choices[random_num].enabled = not is_enable

    def mutate_bias(self, genome):
        """
        Adds a connection between the bias node and one other non-input
        neuron with a default weight of 1. Adds bias node connections
        based on the mutation rate
        """

        mutation_rate = genome.mutation_rates.bias

        while mutation_rate > 0:
            if random.random() < mutation_rate:

                bias_node = genome.num_inputs - 1
                random_num = random.randint(genome.num_inputs, genome.max_neuron - 1)

                new_gene = Gene(random_num, bias_node, (random.random * 4) - 2,
                                True, self.pool.get_innovation())
                genome.genes.append(new_gene)

    def create_population(self, is_connected):
        """
        Creates a population that is structured based on the model variables.
        This initial population can either be connected or not based on is_connected.

        :param is_connected:    Whether or not the genomes should be connected or not
        """

        pop_size = self.model_constants.population_size
        input_size = self.model_constants.inputs
        output_size = self.model_constants.outputs

        for genome_num in range(0, pop_size):
            if is_connected:
                self.population.append(self.basic_genome_connected(input_size, output_size))
            else:
                self.population.append(self.basic_genome(input_size, output_size))

    # TODO UPDATE FITNESS SCORING TO ACCOUNT FOR STALENESS
    def speciate_population(self, species_list):
        """
        Creates a new pool that contains the speciated population from
        the last generation

        :param species_list:    The list of species in the previous generation
        :return:                The new pool of speciated genomes
        """

        compat_diff = self.speciation_values.compat_constant
        species_reps = []
        species_pop = []

        for species in species_list:
            species_pop.append([])
            num_genomes = len(species.genomes)
            species_reps.append(species.genomes[random.randint(0, num_genomes - 1)])

        # Run through the population, while assigning the individuals to species
        for genome in self.population:
            is_assigned = False
            cur_species = 0

            # Checks the list of species to see if the current genome is compatible with any
            # of them. If it is then add genome to the species
            while (not is_assigned) & (cur_species < len(species_reps)):
                cur_compat = self.compatibility_difference(genome, species_reps[cur_species])

                if cur_compat < compat_diff:
                    is_assigned = True
                    species_pop[cur_species].append(genome)

                cur_species += 1

            # If the genome does not find a species, then create a species for this genome
            if not is_assigned:
                species_pop.append([genome])
                species_reps.append(genome)

        new_species_list = []

        for cur_species in range(0, len(species_list)):
            new_specie = Species()
            new_specie.staleness = species_list[cur_species].stalentess
            new_specie.genomes = species_pop[cur_species]

        for cur_species in range(len(species_list), len(species_reps)):
            new_specie = Species()
            new_specie.staleness = 0
            new_specie.genomes = species_pop[cur_species]
            new_species_list.append(new_specie)

        pool = Pool(new_species_list, self.pool.generation, self.pool.innovation, 0, 0, 0)

        return pool

    # TODO Create proof as to why normalizing on Total number of genes is better
    # than normalizing on number of genes in the larger genome
    def compatibility_difference(self, genome1, genome2):
        """
        Finds the compatibility difference between two genomes.

        :param genome1:     The first genome to be compared
        :param genome2:     The second genome to be compared
        :return:            Returns the compatibility difference
        """

        max_innovation = self.pool.innovation

        genome1_genes = [[] for _ in range(0, max_innovation)]
        genome2_genes = [[] for _ in range(0, max_innovation)]

        disjoint_genes = 0
        excess_genes = 0
        conjoint_genes = 0
        weight_diff_total = 0

        genome1_max_innov = genome1.max_innovation
        genome2_max_innov = genome2.max_innovation

        # Sorts genes into easily accessed lists where each gene is at the
        # index of its innovation number
        for gene in genome1.genes:
            genome1_genes[gene.innovation] = gene

        # Sorts genes into easily accessed lists where each gene is at the
        # index of its innovation number. While going through the list of
        # genes in the genome we check for disjoint and excess genes
        for gene in genome2.genes:
            genome2_genes[gene.innovation] = gene
            if genome1_genes[gene.innovation]:
                if gene.innovation > genome1_max_innov:
                    excess_genes += 1
                else:
                    disjoint_genes += 1

        # Go through genome1's genes and find count the number of conjoint,
        # disjoint, and excess genes. As well as the total weight difference
        # between conjoint genes
        for gene in genome1.genes:
            if genome2_genes[gene.innovation]:
                if gene.innovation > genome2_max_innov:
                    excess_genes += 1
                else:
                    disjoint_genes += 1
            else:
                conjoint_genes += 1
                weight_diff_total += abs(gene.weight - genome2_genes[gene.innovation].weight)

        # The rest is just computing the compatibility difference based on values we have
        # already found. We normalize the values of disjoint_genes and excess_genes, in
        # order to account for varying sizes of genomes
        weight_diff_avg = weight_diff_total / conjoint_genes
        compat_values = self.speciation_values
        num_genes = len(genome1.genes) + len(genome2.genes)
        disjoint_genes /= num_genes
        excess_genes /= num_genes
        compatibility_diff = ((disjoint_genes * compat_values.disjoint_coeff) +
                              (excess_genes * compat_values.excess_coeff) +
                              (weight_diff_avg * compat_values.weight_coeff))

        return compatibility_diff

    # TODO Add random gene selection between genomes with equal fitness
    def crossover_genes(self, genome1, genome2):
        """
        Combines two genomes into a new genome with all of the superior fitnesses'
        genes and the excess and disjoint genes of the genome with the lower fitness

        :param genome1:     The first genome in the crossover
        :param genome2:     The second genome in the crossover
        :return:            A new genome that is a crossover of the two genome parameters
        """

        if genome1.fitness < genome2.fitness:
            temp = genome1
            genome1 = genome2
            genome2 = temp

        new_genome = self.copy_genome(genome1)
        new_genome.num_inputs = genome1.num_inputs
        new_genome.num_outputs = genome1.num_outputs
        new_genome.max_neuron = max(genome1.max_neuron, genome2.max_neuron)
        new_genome.mutation_rates = self.copy_mutation_rates(genome1.mutation_rates)

        innovations = [[] in range(0, self.pool.innovation)]

        for gene in genome1.genes:
            innovations[gene.innovation] = gene

        for gene in genome2.genes:
            # If the innovation does not exist in genome1 then it is either an excess
            # or disjoint gene and should be added to the new_gene
            if not innovations[gene.innovation]:
                new_genome.add_gene(gene)

        return new_genome

    # TODO
    def run_simulation(self, num_generations):

        self.create_population(False)

        for generation in range(0, num_generations):
            self.run_generation()

    # TODO
    def run_generation(self):

        if not self.population:
            raise NotImplementedError("Population does not exist")


    def selection_crossover(self):
        """
        Completes the selection of genomes to be bred and then creates the child
        by crossover. The genomes selected to be in crossover are selected based
        on the Fitness Proportionate Selection model.

        :return:    The genome created by crossover
        """

        species = self.pool.species
        interspecies_mating_rate = self.model_rates.interspecies_mating_rate

        if (len(species) > 1) & (random.random() <= interspecies_mating_rate):

            species_chosen = random.sample(range(0,len(species)), 2)
            genome1 = self.genome_selection(species[species_chosen[0]])
            genome2 = self.genome_selection(species[species_chosen[1]])
        else:
            species_chosen = random.randint(0,len(species) - 1)
            genome1 = self.genome_selection(species[species_chosen])
            genome2 = self.genome_selection(species[species_chosen])
        return self.crossover_genes(genome1, genome2)

    def genome_selection(self, species):
        """
        Uses Fitness Proportionate Selection to select genomes from a
        species to be bred

        :param species: The species a genome will be selected from
        :return:        The genome to be bred
        """

        fitness = []
        fitness_total = 0

        for genome in species.genomes:
            fitness.append(genome.fitness)
            fitness_total += genome.fitness

        random_num = random.random()
        cdf = 0         # cumulative distribution function over fitness

        for cur_genome in range(0, len(fitness)):
            cdf += (fitness[cur_genome]/fitness_total)
            if cdf <= random_num:
                return species.genomes[cur_genome]

        return species.genomes[len(fitness) - 1]

    def score_fitness(self):
        """
        Scores all the genomes fitnesses and then calculates the shared fitnes
        """

        self.score_genomes()

        # Goes through each species and updates the shared fitness of the genomes
        # and checks for a new max fitness for the species
        for species in self.pool.species:
            species_total_fitness = 0
            size_species = len(species.genomes)

            for genome in species.genomes:
                genome.shared_fitness = genome.fitness / size_species
                if genome.shared_fitness > species.top_fitness:
                    species.top_fitness = genome.shared_fitness
                species_total_fitness += genome.shared_fitness

            species.average_fitness = (species_total_fitness / size_species)

    # TODO DIVIDE INTO MODELCONSTANTS.X NUMBER OF GAMES OF MUDELCONSTANT.PARALLEL_EVAL
    # TODO NETWORKS AND EVALUATE EACH NETWORKS FITNESS BASED ON GAMES
    def score_genomes(self):
        """
        Splits all of the genomes into games based on what GameConstants have been decided
        and then plays those games individually
        """

        game_division = []
        games = []
        pop_len = len(self.population)

        # A list of randomly ordered elements from 0 to the length of the population - 1
        # There are "game_per_genome" of these lists created and put into one list
        for _ in range(0, len(self.model_constants.games_per_genome)):
            game_division.append(random.sample(range(0, pop_len, pop_len)))

        # Reorders the list of lists of numbers so that now we have a list of "games"
        # where a "game" is a list of the index of players of the game in the population
        # Note: This algorithm allows two of the same player to be in the same game
        for game_iter in range(0, len(game_division)):
            players = []
            for cur_player in range(0, self.model_constants.parallel_evals):
                players.append(game_division[cur_player][game_iter])
            games.append(players)

        self.play_games(games)

    def play_games(self, games):

        game_model = self.game_model