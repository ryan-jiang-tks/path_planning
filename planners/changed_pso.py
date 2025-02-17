import numpy as np

class MatrixPSO:
    def __init__(self, objective_function, dimensions, bounds=None, num_particles=30, 
                 iterations=100, initial_positions=None, inertia=0.5, 
                 cognitive=1.5, social=1.5, inertia_decay=0.99):
        self.objective_function = objective_function
        self.dimensions = dimensions
        self.num_particles = num_particles
        self.iterations = iterations
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.inertia_decay = inertia_decay
        
        # Set bounds as numpy array
        if bounds is None:
            self.bounds = np.array([(1, 29)] * dimensions)
        else:
            self.bounds = np.array(bounds)
            
        # Initialize matrices for particle states
        # Shape: (num_particles, dimensions)
        if initial_positions is not None:
            num_init = min(len(initial_positions), num_particles)
            self.positions = np.zeros((num_particles, dimensions))
            self.positions[:num_init] = initial_positions[:num_init]
            self.positions[num_init:] = np.random.uniform(
                self.bounds[:, 0], 
                self.bounds[:, 1], 
                (num_particles - num_init, dimensions)
            )
        else:
            self.positions = np.random.uniform(
                self.bounds[:, 0], 
                self.bounds[:, 1], 
                (num_particles, dimensions)
            )

        # Initialize velocities as fraction of bounds range
        bounds_range = self.bounds[:, 1] - self.bounds[:, 0]
        self.velocities = np.random.uniform(-1, 1, (num_particles, dimensions)) * bounds_range * 0.1
        
        # Initialize best positions and values
        self.best_positions = self.positions.copy()
        self.best_values = np.full(num_particles, float('inf'))
        
        # Global best
        self.global_best_position = None
        self.global_best_value = float('inf')
        
        # History
        self.iteration_best_positions = []

    def update_velocities(self):
        """Update all particles' velocities using matrix operations"""
        r1 = np.random.random((self.num_particles, 1))
        r2 = np.random.random((self.num_particles, 1))
        
        cognitive_term = self.cognitive * r1 * (self.best_positions - self.positions)
        social_term = self.social * r2 * (self.global_best_position - self.positions)
        
        self.velocities = (self.inertia * self.velocities + 
                          cognitive_term + 
                          social_term)

    def update_positions(self):
        """Update all particles' positions using matrix operations"""
        self.positions += self.velocities
        # Bound positions using broadcasting
        self.positions = np.clip(self.positions, self.bounds[:, 0], self.bounds[:, 1])

    def evaluate_particles(self):
        """Evaluate all particles at once if possible"""
        values = np.array([self.objective_function(p) for p in self.positions])
        
        # Update personal bests
        better_indices = values < self.best_values
        self.best_positions[better_indices] = self.positions[better_indices]
        self.best_values[better_indices] = values[better_indices]
        
        # Update global best
        current_best_idx = np.argmin(values)
        if values[current_best_idx] < self.global_best_value:
            self.global_best_value = values[current_best_idx]
            self.global_best_position = self.positions[current_best_idx].copy()
            
        return values, current_best_idx

    def optimize(self):
        """Optimize using matrix operations"""
        self.iteration_best_positions = []
        
        for _ in range(self.iterations):
            # Evaluate particles and update bests
            values, current_best_idx = self.evaluate_particles()
            
            # Store iteration history
            self.iteration_best_positions.append(
                (self.positions[current_best_idx].copy(), values[current_best_idx])
            )
            
            # Update velocities and positions
            self.update_velocities()
            self.update_positions()
            
            # Decay inertia
            self.inertia *= self.inertia_decay
        
        return self.global_best_position, self.global_best_value

    def get_optimization_history(self):
        return self.iteration_best_positions

# # Example vectorized objective function
# def objective_function(x):
#     return np.sum(np.square(x))
