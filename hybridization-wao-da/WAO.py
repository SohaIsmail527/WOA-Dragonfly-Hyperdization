import numpy as np
from copy import copy
from HydrogenAtom import HydrogenAtom

class WAO:
    def __init__(self, fitness_func, dimensions=10, sample_size=100, boundaries=(-100, 100), iterations=10, fitness_threshold = 500):
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
        # self.initial_population = self.generation_population()

    def generation_population(self):
        """
        Generates a random sample within specified boundaries.
        Returns:
        - A numpy array of shape (sample_size, dimensions) with random samples.
        """
        H_atoms = []
        low, high = self.boundaries
        # Generate the random sample using uniform distribution
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
                        localOPT = min([H_atom, other_atom], key= lambda h: h.fitness)
                        # update position
                        H_atom.position = localOPT.position
                        other_atom.position = localOPT.position
                        # update fitness
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
        """
        Generates a random vector around a given vector within specified bounds.

        Parameters:
        - input_vector: numpy array, the original vector.
        - variation_range: float or numpy array, the maximum variation for each dimension.
                        If a single float, it applies the same range to all dimensions.
                        If an array, it specifies the range for each dimension.

        Returns:
        - A numpy array representing the random vector.
        """
        # Ensure variation_range is an array if a single value is provided
        if np.isscalar(variation_range):
            variation_range = np.full_like(input_vector, variation_range)

        variations = np.random.uniform(-variation_range, variation_range, size=input_vector.shape)
        random_vector = input_vector + variations
        return random_vector

    def wao_runner(self,):
        t_state_pop = self.generation_population()
        generation_vals = []
        for t in range(self.iterations):
          # print("BEFORE*", len(t_state_pop), [k.atom_id for k in t_state_pop])
          bonded_population = self.activate_bonding(t_state_pop)
          # print("AFTER*", len(bonded_population), [k.atom_id for k in bonded_population])

          Afit = self.get_Afit(bonded_population)
          LGfit = self.get_LGfit(bonded_population)
          t_state_pop = []
          for h in bonded_population:
            if h.fitness > Afit:
              # print("hi")
              # check bonding
              if h.bonded_atoms:
                # break bonding
                # print(f"BONDED: {h.atom_id}: {len(h.bonded_atoms)}: {[k.atom_id for k in h.bonded_atoms]}")
                for other_atom in h.bonded_atoms:
                  other_atom.update_position(self.generate_random_vector_around(LGfit.position))
                  t_state_pop.append(other_atom)
                h.bonded_atoms = []
              h.update_position(self.generate_random_vector_around(LGfit.position))
              t_state_pop.append(h)
            else:
              t_state_pop.append(h)
          # print(t, LGfit.fitness, len(LGfit.bonded_atoms))
          generation_vals.append(LGfit.fitness)
        best = self.get_LGfit(t_state_pop)
        return generation_vals, best.position, best.fitness
