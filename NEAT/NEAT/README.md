
## Explanation of Hyperparameters

* model_constants: The constants associated with the neural network model
  * inputs:         The number of input nodes
  * outputs:        The number of output nodes
  * population_size:The size of the population during a generation
  * parallel_evals: The number of games going on at once
  * games_per_genome:The number of games each genome plays during a generation
  * name:           The name of this specific model. Used for saving the model
    
* game_generator:  The method that evaluates a neural networks fitness
  * players_per_game:   The number of networks being evaluated at the same time
  * evals_per_game:     The number of evaluations on a network each game
    
* gen_rates:       The different rates used for generational building
  * child_rate:      The percentage of bred children that make it to the next generation
  * mutated_rate:    The percentage of mutated genomes from the original generation that
                     are brought into the next generation
  * top_x_species:   The number of top genomes from each species with greater than
                     top_x_min_size that are brought into the next round
  * top_x_min_size:  The number of genomes that need to be in a species to guarantee that
                     it is eligible to give its top_x genomes to the next generation
  * cull_rate:       The percentage of genomes that will be removed from the pool before
                     the creation of the next generation begins
  * passthrough_rate:The percentage of genomes that will pass through to the next generation
                     undergoing only mutation
                     
* mutation_rates:  The rates that the neural networks originally get
                        mutated at
  * connection:     The likelihood of a mutation where a connection is added
                    between two pre-existing neurons
  * link:           The likelihood of a mutation where a connections weight
                    is either perturbed or randomly chosen
  * bias:           The likelihood of a mutation where a connection from the
                    bias node to a non-input node is made
  * node:           The likelihood of a mutation where a connection pair
                    (out, into) is disabled, a node 'new' is created, and
                    two connection pairs are created (out, new) and (new, into)
  * enable:         The likelihood of a mutation where a certain gene is
                    enabled.
  * disable:        The likelihood of a mutation where a certain gene is
                    disabled.
  * step:           The maximum change in either direction for the weight of
                    a gene if it is being perturbed     
                    
* speciation_values:   The coefficients for the similarity function
                        as well as the compatibility constant
  * disjoint_coeff:     The coefficient for the penalty on disjoint genes
  * excess_coeff:       The coefficient for the penalty on excess genes
  * weight_coeff:       The coefficient for the penalty on the difference
                        in weights between two genes of the same innovation
  * compat_constant:    The constant for the allowed compatibility difference
                        between genomes of the same species
                        
* model_rates:     The rates of change that the learning model forces
                        on the genomes it maintains
  * inputs:         The number of input nodes
  * outputs:        The number of output nodes
  * population_size:The size of the population during a generation
  * parallel_evals: The number of games going on at once
  * games_per_genome:The number of games each genome plays during a generation
  * name:           The name of this specific model. Used for saving the model
    
* game_constants:  The constants for games to be made
  * players_per_game:   The number of networks being evaluated at the same time
  * evals_per_game:     The number of evaluations on a network each game
