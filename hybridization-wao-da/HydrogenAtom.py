import numpy as np 

class HydrogenAtom():
    def __init__(self, atom_id, position, fitness_func, bounds):
        self.atom_id = atom_id
        self.position = np.array(position)
        self.bounds = bounds
        self.fitness_func = fitness_func
        self.fitness = fitness_func(self.position)
        self.bonded_atoms = []  # List of atoms this one is bonded with

    def update_position(self, new_position):
        # Update the atom's position and fitness
        self.position = np.clip(np.array(new_position), *self.bounds)  # Ensure new position is within bounds
        self.fitness = self.fitness_func(self.position)

    def try_bonding(self, other_atom, fitness_threshold):
        # Attempt to bond with another atom based on fitness
        if abs(self.fitness - other_atom.fitness) < fitness_threshold:
            return True
        return False

    def __repr__(self):
        # String representation for debugging
        return f"HydrogenAtom(position={self.position}, fitness={self.fitness})"