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
     the advantage of saving space, as an adjacency matrix costs O(V^2) whereas an
     adjacency list costs O(V*E). On top of that searching for all the edges off of
     a node takes best, average, and worst case O(E), whereas the double adjacency
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
# TODO FOR FUTURE: REMOVE STALE SPECIES AND REPLACE WITH BASIC GENOMES
# TO FACILITATE RANDOM RESTARTING WHEN THE POPULATION STALES OUT
# TODO Make networks updatable so that they are not constantly recreated
# TODO Networks can remake connections that have been disabled, this is a bug
# TODO Redudant holding of topological_order in genome and network this may be a feature
#           or a bug, more deliberation on that topic needed
import math
import random
import copy


# **********************************************************************************
# ***************************** Management Functions *******************************
# **********************************************************************************

def topological_sort(genome):
    """
    A topological sort (using Kahn's Algorithm) to create an order of
    the poset that will be used to ensure the directed acyclic graph
    stays acyclic when adding edges. The design decisions for this
    algorithm are explained in documentation point 3.

    :param genome:   A list of genes that represent the genome
    :return:        A list representing the topological order
                    of the nodes in the neural network
    """

    copy_genes = []
    for gene in genome.genes:
        copy_genes.append(copy_gene(gene))

    adj_list = [[] for _ in range(0, len(genome.topological_order))]

    # This adjacency list is atypical in that at each index i
    # there is a list of indices such that each element j
    # in that list denotes an edge (j,i) in the graph.
    # (As opposed to the typical (i,j) representation)
    adj_list_rev = [[] for _ in range(0, len(genome.topological_order))]

    sorted_list = [x for x in range(0, genome.num_inputs + 1)]
    cur_list = []
    next_list = []

    # Fills the two adjacency lists with the edges from the genes
    for gene in copy_genes:
        # We already know the ordering of the input, bias, and output nodes
        # so links to them do not provide insight into the topological ordering
        if ((not gene.out <= genome.num_inputs + genome.num_outputs) &
                (not gene.into <= genome.num_inputs + genome.num_outputs)):
            adj_list_rev[gene.into].append(gene.out)
            adj_list[gene.out].append(gene.into)

    # Creates the initial list from the nodes with no incoming edges
    # There is required to be at least one or the neural network is invalid
    for node_num in range(0, len(genome.topological_order)):
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

    # Add the output nodes to the ordering
    sorted_list.extend([y for y in range(genome.num_inputs + 1,
                                         genome.num_inputs + genome.num_outputs + 1)])
    return sorted_list


# TODO Fix overlap between this and internal function
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

# **********************************************************************************
# ***************************** Class Definitions*** *******************************
# **********************************************************************************


class Pool:
    """
    The complete generational pool of species and data about it

    species:        The list of species in this pool
    generation:     The number of iterations of generations that
                    this pool has already undergone
    innovation:     The next unused value to be given to an innovation
    node_innovation:The next unused value to be given to a node innovation
    current_species:The current species being evaluated
    current_genome: The current genome being evaluated
    max_fitness:    The maximum fitness for any genome in the pool
    node_history:   List of ((node_innov1, node_innov2), new_innov) where
                    node_innov1 is the out of node and node_innov2 is the into node
                    where we placed a new node of new_innov
    gene_history:   List of ((node_innov1, node_innov2), new_innov) where
                    node_innov1 is the out of node and node_innov2 is the into node
                    where we placed a connection of new_innov
    """

    def __init__(self, species=None, generation=0, innovation=0, node_innovation=0,
                 current_species=0, current_genome=0, max_fitness=0, node_history=None,
                 gene_history=None):
        if species is None:
            species = []
        if node_history is None:
            node_history = {}
        if gene_history is None:
            gene_history = {}
        self.species = species
        self.generation = generation
        self.innovation = innovation
        self.node_innovation = node_innovation
        self.current_species = current_species
        self.current_genome = current_genome
        self.max_fitness = max_fitness
        self.node_history = node_history
        self.gene_history = gene_history

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

    def get_node_innovation(self):
        """
        Increments and returns the node innovation number

        :return:    The current node innovation number
        """

        to_return = self.node_innovation
        self.node_innovation += 1
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
    global_rank:    The current rank of the genome
    topological_order:A list that represents the topological order
                    of the neurons
    """

    def __init__(self, genes=None, fitness=0, shared_fitness=0, network=None,
                 num_inputs=0, num_outputs=0, global_rank=0,
                 mutation_rates=None, topological_order=None):
        if network is None:
            network = []
        # If there is no starting hidden structure then set number of neurons to
        # input+output+bias and create initial topology
        if genes is None:
            genes = []
            if topological_order is None:
                topological_order = random.sample(range(0, num_inputs + 1), num_inputs + 1)
                outputs = random.sample(range(num_inputs + 1,
                                              num_inputs + num_outputs + 1), num_outputs)
                topological_order.extend(outputs)
        if mutation_rates is None:
            mutation_rates = MutationRates()

        self.genes = genes
        self.fitness = fitness
        self.shared_fitness = shared_fitness
        self.network = network
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.global_rank = global_rank
        self.mutation_rates = mutation_rates

        # If there is an initial structure, but no topological order then we need to find one
        if (topological_order is None) & (self.genes == []):
            topological_order = topological_sort(self)
        self.topological_order = topological_order

    def crossover_add_gene(self, gene):
        """
        Adds the gene into the genome

        :param gene:    The gene to be added
        """

        self.genes.append(gene)

    #TODO Can create connection from a node to itself
    def mutate_connection(self, pool):
        """
        Adds a gene to the genome randomly and changes the necessary
        parameters associated
        """

        random_list = copy.deepcopy(self.topological_order)
        random_index1 = random.randint(0, len(random_list) - 1)
        random_node1 = random_list[random_index1]

        inputs = self.topological_order[: self.num_inputs + 1]
        outputs = self.topological_order[len(self.topological_order) - self.num_outputs:]

        index2_options = [x for x in range(0, len(random_list))]
        # If we chose a input or bias, don't let us do that for the second choice
        if inputs.__contains__(random_node1):
            index2_options = index2_options[self.num_inputs + 1:]
        # If we chose output, don't let us do that for the second choice
        elif outputs.__contains__(random_node1):
            index2_options = index2_options[: len(index2_options) - self.num_outputs]
        else:
            index2_options.remove(random_index1)

        random_index2 = index2_options[random.randint(0, len(index2_options) - 1)]
        random_node2 = random_list[random_index2]

        # Ensures there are no recurrent connections, by forcing node2 to have
        # the greater index
        if random_index1 > random_index2:
            temp = random_node1
            random_node1 = random_node2
            random_node2 = temp

        # Checks if this gene has been mutated before
        does_gene_exist = pool.gene_history.get((random_node2, random_node1))

        if does_gene_exist is None:
            innovation_num = pool.get_innovation()
        else:
            innovation_num = does_gene_exist
            # Checks to ensure this gene does not already exist in this genome
            for gene in self.genes:
                if gene.innovation == innovation_num:
                    return

        pool.gene_history[(random_node2, random_node1)] = innovation_num

        random_num = (random.random() * 4) - 2
        new_gene = Gene(random_node2, random_node1, random_num, True, innovation_num)
        self.genes.append(new_gene)

    # TODO Add a "shuffle topology" where we recalculate the topology to change poset
    def mutate_node(self, pool):
        """
        Adds two genes to the genomes where one used to be. Updates class values as
        needed
        """

        enabled_genes = []
        # Compiles all of the genes that are enabled and not attached to the bias node
        for gene in self.genes:
            if gene.enabled & (not gene.out == self.num_inputs):
                enabled_genes.append(gene)

        # Checks to make sure there are enabled genes
        if len(enabled_genes) <= 0:
            return

        old_gene = enabled_genes[random.randint(0, len(enabled_genes) - 1)]

        does_node_exist = pool.node_history.get((old_gene.into, old_gene.out))

        # Checks for this node in the pool's history of node mutations
        # and adds to the pool's history if necessary
        if does_node_exist is None:
            new_node_innov = pool.get_node_innovation()
            pool.node_history[(old_gene.into, old_gene.out)] = new_node_innov
            innovation_num1 = pool.get_innovation()
            innovation_num2 = pool.get_innovation()
            pool.gene_history[(old_gene.out, new_node_innov)] = innovation_num1
            pool.gene_history[(new_node_innov, old_gene.into)] = innovation_num2
        else:
            new_node_innov = does_node_exist
            first_gene = pool.gene_history[(old_gene.out, new_node_innov)]
            second_gene = pool.gene_history[(new_node_innov, old_gene.into)]
            if first_gene is None:
                innovation_num1 = pool.get_innovation()
                pool.gene_history[(new_node_innov,
                                        old_gene.out)] = innovation_num1
            else:
                innovation_num1 = first_gene
            if second_gene is None:
                innovation_num2 = pool.get_innovation()
                pool.gene_history[(old_gene.into,
                                        new_node_innov)] = innovation_num2
            else:
                innovation_num2 = second_gene

        # The connection into the new node has a weight of 1 while the connection into
        # the old_gene's into node receives the old_gene's weight.
        # This minimizes the immediate effect on the networks fitness
        gene_split1 = Gene(new_node_innov, old_gene.out, 1.0, True, innovation_num1)
        gene_split2 = Gene(old_gene.into, new_node_innov, old_gene.weight,
                           True, innovation_num2)

        # Markers for the end of the inputs and the start of the outputs
        not_before = self.num_inputs + 1
        not_after = len(self.topological_order) - self.num_outputs

        # Checks for further restrictions on the topological placement of our new node
        # based on the endpoints of the connection being mutated
        out_index = self.topological_order.index(old_gene.out) + 1
        into_index = self.topological_order.index(old_gene.into)
        if out_index > not_before:
            not_before = out_index
        if into_index < not_after:
            not_after = into_index

        placement = random.randint(not_before, not_after)

        self.topological_order.insert(placement, new_node_innov)

        old_gene.enabled = False
        self.genes.append(gene_split1)
        self.genes.append(gene_split2)

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
        to_return += "," + str(len(self.topological_order)) + "," + str(self.global_rank)
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
    A set of neurons that make up a network. This is a static implementation.
    Adding neurons to an already created Network is not currently supported,
    nor are there plans to support it in the future

    num_inputs:     The number of neurons that are in the input layer
    num_outputs:    The number of neurons that are in the output layer
    neurons:        List of neurons in the neural network. The list is
                    ordered from first to last topological order except
                    for the output layer which immediately follows the
                    input layer
    topological_order:  The topological ordering of the nodes in the network
    """

    def __init__(self, num_inputs=0, num_outputs=0, neurons=None, topological_order=None):
        if neurons is None:
            neurons = []
        elif topological_order is None:
            error_msg = "Automatic topology order finding is not currently supported:"
            error_msg += "\n\tPlease explicitly assign a topological order"
            raise NotImplementedError(error_msg)
        if topological_order is None:
            topological_order = []
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.neurons = neurons
        self.topological_order = topological_order


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

    def __init__(self, label, incoming_neurons=None, weights=None, value=0.0):
        if incoming_neurons is None:
            incoming_neurons = []
        if weights is None:
            weights = []
        self.label = label
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
                    enabled.
    disable:        The likelihood of a mutation where a certain gene is
                    disabled.
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


# TODO Utilize the parallel_evals variable to run multiple games at the same time
class ModelConstants:
    """
    Constants that describe the model

    inputs:         The number of input nodes
    outputs:        The number of output nodes
    population_size:The size of the population during a generation
    parallel_evals: The number of games going on at once
    games_per_genome:The number of games each genome plays during a generation
    """
    def __init__(self, inputs, outputs, population_size, parallel_evals, games_per_genome):
        self.inputs = inputs
        self.outputs = outputs
        self.population_size = population_size
        self.parallel_evals = parallel_evals
        self.games_per_genome = games_per_genome


class GameConstants:
    """
    Constants that describe how the games should be played

    players_per_game : The number of networks being evaluated at the same time
    evals_per_game   : The number of evaluations on a network each game
    """
    def __init__(self, players_per_game, hands_per_game):
        self.players_per_game = players_per_game
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
    interspecies_mating_rate: The chance that when breeding occurs it will
                    be between two individuals of the different species
    """
    def __init__(self, perturb_rate, interspecies_mating_rate):
        self.perturb_rate = perturb_rate
        self.interspecies_mating_rate = interspecies_mating_rate


class GenerationRates:
    """
    A collection of rates on how the population changes from generation to generation

    child_rate:     The percentage of bred children that make it to the next generation
    mutated_rate:   The percentage of mutated genomes from the original generation that
                    are brought into the next generation
    top_x_species:  The number of top genomes from each species with greater than
                    top_x_min_size that are brought into the next round
    top_x_min_size: The number of genomes that need to be in a species to guarantee that
                    it is eligible to give its top_x genomes to the next generation
    cull_rate:      The percentage of genomes that will be removed from the pool before
                    the creation of the next generation begins
    Rates in NEAT Paper:
    child_rate = .75
    mutated_rate = .25
    top_x_species = 1 "The champion of each species with more than five networks was brought
                       into the next generation unchanged"
    top_x_min_size = 5 This follows directly from the above statement
    cull_rate = No mention in the paper however, SethBling uses a value of .5


    child_rate and mutated_rate have been removed and become an implied value. I have
    left the documentation for them here, in case they need to be added back as a
    forced value
    """
    def __init__(self, top_x_species, top_x_min_size, cull_rate):
        self.top_x_species = top_x_species
        self.top_x_min_size = top_x_min_size
        self.cull_rate = cull_rate


class LearningModel:
    def __init__(self, model_constants, game_generator, gen_rates,
                 mutation_rates, speciation_values, model_rates, game_constants):
        """
        Saves the initial parameters

        :param model_constants: The constants associated with the neural network model
        :param game_generator:  The method that evaluates a neural networks fitness
        :param gen_rates:       The different rates used for generational building
        :param mutation_rates:  The rates that the neural networks originally get
                                mutated at
        :param speciation_values:   The coefficients for the similarity function
                                as well as the compatibility constant
        :param model_rates:     The rates of change that the learning model forces
                                on the genomes it maintains
        :param game_constants:  The constants for games to be made
        """

        if mutation_rates is None:
            mutation_rates = MutationRates()
        self.model_constants = model_constants
        self.game_generator = game_generator
        self.gen_rates = gen_rates
        self.mutation_rates = mutation_rates
        self.speciation_values = speciation_values
        self.model_rates = model_rates
        self.game_constants = game_constants
        innovation_start = model_constants.inputs + model_constants.outputs + 1
        self.pool = Pool(innovation=innovation_start)
        self.population = []

    # **********************************************************************************
    # ***************************** Save and Load Tools ********************************
    # TODO DEPRECATED DEPRECATED DEPRECATED DEPRECATED DEPRECATED DEPRECATED DEPRECATED
    # TODO DEPRECATED DEPRECATED DEPRECATED DEPRECATED DEPRECATED DEPRECATED DEPRECATED
    # TODO DEPRECATED DEPRECATED DEPRECATED DEPRECATED DEPRECATED DEPRECATED DEPRECATED
    # TODO DEPRECATED DEPRECATED DEPRECATED DEPRECATED DEPRECATED DEPRECATED DEPRECATED
    # TODO need to rewrite based on new structure of classes (Pool, Genome, Genes)
    # **********************************************************************************

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
            cur_genome.topological_order = topological_sort(struct_contents)

            max_innov = 0
            # Finds the max innovation number from the genes in the genome
            for gene in cur_genome.genes:
                if gene.innovation > max_innov:
                    max_innov = gene.innovation

            cur_item_index = content_bounds[1] + 2
            end_item_index = cur_item_index + file_contents[cur_item_index: file_len].find(";")

            list_items = file_contents[cur_item_index: end_item_index].split(",")

            cur_genome.fitness = int(list_items.pop(0))
            cur_genome.shared_fitness = int(list_items.pop(0))
            cur_genome.num_inputs = int(list_items.pop(0))
            cur_genome.num_outputs = int(list_items.pop(0))
            list_items.pop(0)
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

    # **********************************************************************************
    # ******************************** Managing Library ********************************
    # **********************************************************************************

    def set_pool(self, value):
        self.pool = value

    def set_mutation_rates(self, value):
        self.mutation_rates = value

    @staticmethod
    def sigmoid(value):
        """
        Returns the sigmoid of 'value'
        :param value:   Value for sigmoid function to
        :return:        Return sigmoid of value
        """
        denom = 1 + math.exp(-4.9*value)
        return (2/denom)-1

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

        copy_mut = MutationRates(mut_rates.connection, mut_rates.link, mut_rates.bias,
                                 mut_rates.node, mut_rates.enable, mut_rates.disable, mut_rates.step)
        return copy_mut

    def copy_network(self, network):
        """
        Deep copy of 'network'
        :param network:     Network to be copied
        :return:            Returns a deep copy of the mutation rates
        """

        network_copy = Network()
        for neuron in network.neurons:
            network_copy.neurons.append(self.copy_neuron(neuron))
        network_copy = copy.deepcopy(network.topological_order)
        network_copy.num_inputs = network.num_inputs
        network_copy.num_outputs = network.num_outputs
        return network_copy

    def copy_neuron(self, neuron):

        neuron_copy = Neuron()
        neuron_copy.label = neuron.label
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
        num_nodes = len(genome.topological_order)
        neuron_order = genome.topological_order

        neuron_list = []

        # Appends all the neurons to the neuron_list
        for x in range(0, num_nodes):
            neuron_list.append(Neuron(genome.topological_order[x]))

        # Creates connections between neurons based off of genes in the genome
        for gene in genome.genes:
            if gene.enabled:
                cur_neuron = neuron_list[genome.topological_order.index(gene.into)]
                cur_neuron.incoming_neurons.append(gene.out)
                cur_neuron.weights.append(gene.weight)

        # Adds all of the input and output nodes to the order
        for neuron_iter in range(0, genome.num_inputs+genome.num_outputs):
            network.neurons.append(neuron_list[neuron_iter])
            neuron_order.remove(neuron_iter)

        for node in neuron_order:
            network.neurons.append(neuron_list[node])

        network.topological_order = genome.topological_order

        return network

    # TODO TEST EVALUATE_NEURON_VALUES (CURRENTLY UNTESTED
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

        hidden_start = network.num_inputs + network.num_outputs + 1

        # Finds out the value of every next neuron based on previous neurons'
        # values and the structure of the graph
        for hidden_iter in range(hidden_start, len(network.neurons)):
            cur_node = network.neurons[hidden_iter]
            total_value = 0

            for incoming_iter in range(0, len(cur_node.incoming_neurons)):
                cur_connection = network.topological_order.index(cur_node.incoming_neurons[incoming_iter])
                cur_value = network.neurons[cur_connection].value
                total_value += cur_value * cur_node.weights[incoming_iter]

            cur_node.value = self.sigmoid(total_value)

        to_return = []
        # Collecting neural network outputs
        for output_iter in range(network.num_inputs, network.num_inputs + network.num_outputs):
            to_return.append(network.neurons[output_iter])

        return to_return

    def basic_genome(self, inputs, outputs):
        """
        Creates and returns a default genome with no connections

        :param inputs:  The number of input variables for the neural network
        :param outputs: The number of output variables for the neural network
        :return:        A default genome with no connections
        """
        genome = Genome(None, 0, 0, None, inputs + 1, outputs, inputs + outputs + 1,
                        0, 0, self.copy_mutation_rates(self.mutation_rates), None)
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

    # **********************************************************************************
    # ******************************* Mutation Functions *******************************
    # **********************************************************************************

    def mutate_pool(self):
        """
        Mutate all of the genomes that are still in the pool
        """

        pool = self.pool

        for species in pool.species:
            for genome in species.genomes:
                self.mutate(genome)

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
                genome.mutate_connection(self.pool)
            mutation_rate -= 1

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
                genome.mutate_node(self.pool)
            mutation_rate -= 1

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

        if len(choices) > 0:
            random_num = random.randint(0, len(choices))
            choices[random_num].enabled = not is_enable

    # TODO CHANGE MUTATE BIAS TO ACTUALLY WORK
    def mutate_bias(self, genome):
        """
        Adds a connection between the bias node and one other non-input
        neuron with a default weight of 1. Adds bias node connections
        based on the mutation rate
        """

        mutation_rate = genome.mutation_rates.bias

        while mutation_rate > 0:

            if random.random() < mutation_rate:

                choices = genome.topological_order
                choices = choices[genome.num_inputs + 1:]

                for gene in genome.genes:
                    if gene.out == genome.num_inputs:
                        choices.remove(gene.into)

                bias_node = genome.num_inputs - 1
                random_num = random.randint(genome.num_inputs, len(genome.topological_order))
                random_node = genome.topological_order[random_num]

                new_gene = Gene(random_node, bias_node, (random.random() * 4) - 2,
                                True, self.pool.get_innovation())
                genome.genes.append(new_gene)
            mutation_rate -= 1

    # **********************************************************************************
    # ****************************** Speciation Functions ******************************
    # **********************************************************************************

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

        pool = Pool(new_species_list, self.pool.generation, self.pool.innovation,
                    self.pool.node_innovation, 0, 0, 0, self.pool.node_history,
                    self.pool.gene_history)

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

        disjoint_genes = 0
        excess_genes = 0
        conjoint_genes = 0
        weight_diff_total = 0

        max_innov = 0
        for gene in genome1.genes:
            if gene.innovation > max_innov:
                max_innov = gene.innovation
        genome1_max_innov = max_innov

        max_innov = 0
        for gene in genome2.genes:
            if gene.innovation > max_innov:
                max_innov = gene.innovation
        genome2_max_innov = max_innov

        max_size = max(genome1_max_innov, genome2_max_innov)
        genome1_genes = [[] for _ in range(0, max_size)]
        genome2_genes = [[] for _ in range(0, max_size)]

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

        # Go through genome1's genes and count the number of conjoint,
        # disjoint, and excess genes. As well as the total weight difference
        # between conjoint genes
        for gene in genome1.genes:
            if not genome2_genes[gene.innovation]:
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
        if conjoint_genes > 0:
            weight_diff_avg = weight_diff_total / conjoint_genes
        else:
            weight_diff_avg = 0
        compat_values = self.speciation_values
        num_genes = len(genome1.genes) + len(genome2.genes)

        if num_genes == 0:
            compatibility_diff = 0
        else:
            disjoint_genes /= num_genes
            excess_genes /= num_genes
            compatibility_diff = ((disjoint_genes * compat_values.disjoint_coeff) +
                                  (excess_genes * compat_values.excess_coeff) +
                                  (weight_diff_avg * compat_values.weight_coeff))

        return compatibility_diff

    # **********************************************************************************
    # **************************** Outward Facing Functions ****************************
    # **********************************************************************************

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
                new_genome = self.basic_genome_connected(input_size, output_size)
            else:
                new_genome = self.basic_genome(input_size, output_size)
            self.mutate(new_genome)
            self.population.append(new_genome)

    # TODO
    def run_simulation(self, num_generations):

        self.create_population(False)

        for generation in range(0, num_generations):
            self.run_generation()

    # TODO
    def run_generation(self):

        self.pool = self.speciate_population(self.pool.species)
        self.score_genomes()

        self.build_generation()

    # **********************************************************************************
    # ************************** Building the Next Generation **************************
    # **********************************************************************************

    def build_generation(self):
        """
        Completes the culling, breeding, and mutating stages of the NEAT cycle.

        :return:    Returns a list of genomes to be put into the next pool
        """

        gen_size = self.model_constants.population_size
        species = self.pool.species

        top_x_genomes = self.isolate_top_x()

        self.cull_population()

        num_mutated = 0

        # Figures out how many genomes are moving into the next generation
        # after being mutated to see how many children genomes need to be bred
        for specie in species:
            num_mutated += len(specie.genomes)

        num_cross = gen_size - len(top_x_genomes) - num_mutated
        crossed_genomes = []

        # Create a certain number of bred children that fills in the rest of the population
        for cross_iter in range(0, num_cross):
            crossed_genomes.append(self.selection_crossover())

        self.mutate_pool()

        mutated_genomes = []

        # Adds all the mutated genomes to the list
        for specie in species:
            mutated_genomes.extend(specie.genomes)

        to_return = mutated_genomes
        to_return.extend(crossed_genomes)
        to_return.extend(top_x_genomes)

        return to_return

    def isolate_top_x(self):
        """
        Saves the top_x competitors of each species if they meet the requirements
        for minimum amount of genomes to contribute

        :return:     "top_x_species" amount of genomes from each species that has
                     at least "top_x_min_size" genomes in its species
                     (both variables are from GenerationRates)
        """

        top_x_species = self.gen_rates.top_x_species
        top_x_min_size = self.gen_rates.top_x_min_size
        species = self.pool.species

        top_x_genomes = []

        # Retrieves an "top_x_species" amount of genomes from each species that has
        # at least "top_x_min_size" genomes in its species
        for species_iter in range(0, len(self.pool.species)):
            if len(species[species_iter]) >= top_x_min_size:
                genome_list = []
                for genome in species[species_iter]:
                    genome_list.append(genome.shared_fitness, (genome, species_iter))
                sorted_genomes = sorted(genome_list)

                genome_iter = len(sorted_genomes) - 1
                list_len = len(sorted_genomes) - 1
                while (genome_iter > 0) & (list_len - genome_iter < top_x_species):
                    top_x_genomes.append(sorted_genomes[genome_iter])
                    genome_iter -= 1

        deep_top_x_genomes = []

        # Make these deep copies, because we will use the others later
        for shallow_copy in top_x_genomes:
            deep_top_x_genomes(self.copy_genome(shallow_copy))

        return deep_top_x_genomes

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
        new_genome.network = []

        innovations = [[] in range(0, self.pool.innovation)]

        for gene in genome1.genes:
            innovations[gene.innovation] = gene

        for gene in genome2.genes:
            # If the innovation does not exist in genome1 then it is either an excess
            # or disjoint gene and should be added to the new_gene
            if not innovations[gene.innovation]:
                new_genome.crossover_add_gene(gene)

        new_genome.topological_order = topological_sort(new_genome)
        return new_genome

    def cull_population(self):
        """
        Removes the genomes with the lowest shared fitness from the pool. The amount of
        genomes removed depends on the given generational rates
        """

        all_genomes = []

        # Creates a list of tuples of the form (shared fitness, (species, cur_genome))
        # Where cur_genome is the genome that is being evaluated, shared fitness is
        # the shared fitness of that genome, and species is the number of the species
        # that the cur_genome belongs to
        for species_iter in range(0, len(self.pool.species)):
            for genome_iter in range(0, len(self.pool.species[species_iter].genomes)):
                cur_genome = self.pool.species[species_iter].genomes[genome_iter]
                all_genomes.append(cur_genome.shared_fitness, (species_iter, cur_genome))

        # Sort the genomes by shared fitness
        sorted_genomes = sorted(all_genomes)

        to_remove = math.ceil(self.gen_rates.cull_rate * len(sorted_genomes))
        # Removes the amount of genomes specified by the given cull_rate
        for worst_genomes_iter in range(0, to_remove):
            (shared_fitness, (species, cur_genome)) = sorted_genomes[worst_genomes_iter]
            self.pool.species[species].remove(cur_genome)

        # Removes any species with no genomes from the pool
        for specie in self.pool.species:
            if len(specie.genomes) == 0:
                self.pool.species.remove(specie)

    # **********************************************************************************
    # ***************************** Fitness Scoring ************************************
    # **********************************************************************************

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
        for _ in range(0, self.model_constants.games_per_genome):
            game_division.append(random.sample(range(0, pop_len), pop_len))

        # Reorders the list of lists of numbers so that now we have a list of "games"
        # where a "game" is a list of the index of players of the game in the population
        # Note: This algorithm allows two of the same player to be in the same game
        for game_iter in range(0, len(game_division)):
            players = []
            for cur_player in range(0, self.game_constants.players_per_game):
                players.append(game_division[game_iter][cur_player])
            games.append(players)

        self.play_games(games)

        # Calculates the adjusted fitness of the genomes and top fitness and average fitness
        # of each species as a whole
        for specie in self.pool.species:
            top_fitness = 0
            total_fitness = 0
            len_specie = len(specie.genomes)
            for genome in specie.genomes:
                if genome.fitness > top_fitness:
                    top_fitness = genome.fitness
                genome.shared_fitness = genome.fitness / len_specie
                total_fitness += genome.fitness
            specie.top_fitness = top_fitness
            specie.average_fitness = total_fitness / len_specie

    def play_games(self, games):
        """
        Takes every game listed in 'games' and runs the game to completion, at which
        point the fitness of every player involved in the game is updated

        :param games:   A list of games where the games are lists of players to be
                        competing in the games
        """

        game_gen = self.game_generator

        # Creates the networks for the population
        for genome in self.population:
            genome.network = self.create_network(genome)

        # Runs each game by facilitating the information transfer between
        # neural networks and the GameModel
        for game in games:
            cur_game = game_gen.create_game(self.game_constants.players_per_game,
                                            self.game_constants.hands_per_game)
            is_game_done = False

            # Continually facilitate information trade, until the game ends
            while not is_game_done:

                (to_act, nn_inputs) = cur_game.send_inputs()
                this_nn = self.population[game[to_act]].network
                nn_output = self.evaluate_neuron_values(this_nn, nn_inputs)
                is_game_done = (cur_game.receive_outputs(nn_output) == -1)

            game_results = cur_game.send_fitness()

            for game_iter in range(0, len(game)):
                self.population[game[game_iter]].fitness += game_results[game_iter]

        # Reset networks to empty, as the same networks are not likely to be used again
        for genome in self.population:
            genome.network = []
