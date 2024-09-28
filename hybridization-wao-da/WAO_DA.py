import numpy as np
from copy import copy
from HydrogenAtom import HydrogenAtom
from dragonfly_algorithm import dragonfly_algorithm

class WAO_DA:
    def __init__(self,
                 fitness_func,
                 dimensions=10,
                 sample_size=100,
                 boundaries=(-100, 100), 
                 iterations=10, 
                 fitness_threshold=500,
                 da_chance=0.2,
                 da_internal_iters=10):
        """
        Parameters:
        - dimensions: int, the number of dimensions for each sample point.
        - sample_size: int, the total number of sample points to generate.
        - boundaries: tuple, a pair of (low, high) specifying the range of values for each dimension.
        - iterations: the total number of iteration which represents the stop condition
        """
        self.fitness_func = fitness_func
        self.dimensions = dimensions
        self.sample_size = sample_size
        self.boundaries = boundaries
        self.iterations = iterations
        self.fitness_threshold = fitness_threshold
        self.da_chance = da_chance
        self.da_internal_iters = da_internal_iters
        self.opt_algorithm_at_every_genration = [self._choose_opt_algorithm() for algo in range(self.iterations)]
        
    def _choose_opt_algorithm(self):
        return np.random.choice(["DA", "WAO"], p=[self.da_chance, 1-self.da_chance])        
    
    def generation_population(self):
        """
        Generates a random sample within specified boundaries.
        Returns:
        - A numpy array of shape (sample_size, dimensions) with random samples.
        """
        H_atoms = []
        low, high = self.boundaries
        sample = np.random.uniform(low, high, size=(self.sample_size, self.dimensions))
        for i, position in enumerate(sample):
            H_atom = HydrogenAtom(f"atom_{i}", position, self.fitness_func, self.boundaries)
            H_atoms.append(H_atom)
        return H_atoms

    def activate_bonding(self, population):
        init_pop = copy(population)
        bonded_population = []
        consumed_atom_ids = []
        for i, H_atom in enumerate(init_pop):
            if H_atom.atom_id not in consumed_atom_ids:
                for j, other_atom in enumerate(init_pop):
                    if H_atom.try_bonding(other_atom, self.fitness_threshold) \
                        and H_atom.atom_id not in consumed_atom_ids \
                        and other_atom.atom_id not in consumed_atom_ids \
                        and H_atom.atom_id != other_atom.atom_id \
                        and other_atom.atom_id not in [k.atom_id for k in H_atom.bonded_atoms]:
                        localOPT = min([H_atom, other_atom], key=lambda h: h.fitness)
                        H_atom.position = localOPT.position
                        other_atom.position = localOPT.position
                        H_atom.fitness = localOPT.fitness
                        other_atom.fitness = localOPT.fitness
                        H_atom.bonded_atoms.append(other_atom)
                        other_atom.bonded_atoms.append(H_atom)
                        consumed_atom_ids.append(other_atom.atom_id)
                        consumed_atom_ids.append(H_atom.atom_id)
                bonded_population.append(H_atom)
        return bonded_population

    def get_Afit(self, population):
        return np.sum([h.fitness for h in population])/len(population)

    def get_LGfit(self, population):
        return min(population, key=lambda k: k.fitness)

    def generate_random_vector_around(self, input_vector, variation_range=3):
        if np.isscalar(variation_range):
            variation_range = np.full_like(input_vector, variation_range)

        variations = np.random.uniform(-variation_range, variation_range, size=input_vector.shape)
        random_vector = input_vector + variations
        return random_vector
  
    def invoke_da(self, start_init):
        da_parameters = {'size': self.sample_size,
            'min_values': [self.boundaries[0] for i in range(self.dimensions)],
            'max_values': [self.boundaries[-1] for i in range(self.dimensions)],
            'generations': self.da_internal_iters,
            'verbose': False,
            'start_init': start_init
            }
        # print(da_parameters)
        da_evaluation = dragonfly_algorithm(target_function = self.fitness_func, **da_parameters)
        # best_position = da_evaluation[:-1]
        # best_value   = da_evaluation[ -1]
        return da_evaluation
    
    def wao_runner(self):
        t_state_pop = self.generation_population()
        generation_vals = []
        best_da_position = None
        for t in range(self.iterations):
            # SELECT OPT algorithm to do optimization    
            # print(t)        
            best_position_so_far = self.get_LGfit(t_state_pop).position
            # print(f"OPT Algorithm: {self.opt_algorithm_at_every_genration[t]}")
            if self.opt_algorithm_at_every_genration[t] == "DA":
                # print(f"DA")
                if t == 0:
                    start_init = None
                else:
                    start_init = best_position_so_far
                da_eval = self.invoke_da(start_init)
                best_da_val, best_da_position = da_eval[-1], da_eval[:-1]
                # UPDATE the best position
                generation_vals.append(best_da_val)
                best_H_atom = self.get_LGfit(t_state_pop)
                best_H_atom.update_position(best_da_position)
            else:
                # print(f"WAO")
                bonded_population = self.activate_bonding(t_state_pop)
                Afit = self.get_Afit(bonded_population)
                LGfit = self.get_LGfit(bonded_population)
                t_state_pop = []
                for h in bonded_population:
                    if h.fitness > Afit:
                        if h.bonded_atoms:
                            for other_atom in h.bonded_atoms:
                                other_atom.update_position(self.generate_random_vector_around(LGfit.position))
                                t_state_pop.append(other_atom)
                            h.bonded_atoms = []
                        h.update_position(self.generate_random_vector_around(LGfit.position))
                        t_state_pop.append(h)
                    else:
                        t_state_pop.append(h)
                generation_vals.append(LGfit.fitness)
        best = self.get_LGfit(t_state_pop)
        return generation_vals, best.position, best.fitness
