import random

class Particle:
    def __init__(self, dimensions, initial_position=None):
        if initial_position is not None:
            self.position = list(initial_position)
        else:
            self.position = [random.uniform(1, 29) for _ in range(dimensions)]
        self.velocity = [random.uniform(-1, 1) for _ in range(dimensions)]
        self.best_position = list(self.position)
        self.best_value = float('inf')

    def update_velocity(self, global_best_position, inertia, cognitive, social):
        for i in range(len(self.velocity)):
            r1 = random.random()
            r2 = random.random()
            cognitive_velocity = cognitive * r1 * (self.best_position[i] - self.position[i])
            social_velocity = social * r2 * (global_best_position[i] - self.position[i])
            self.velocity[i] = inertia * self.velocity[i] + cognitive_velocity + social_velocity

    def update_position(self):
        for i in range(len(self.position)):
            self.position[i] += self.velocity[i]

class PSO:
    def __init__(self, objective_function, dimensions, num_particles, iterations, 
                 initial_positions=None, inertia=0.5, cognitive=1.5, social=1.5):
        self.objective_function = objective_function
        self.dimensions = dimensions
        self.num_particles = num_particles
        self.iterations = iterations
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social

        # Initialize particles with provided positions if available
        self.particles = []
        for i in range(num_particles):
            if initial_positions and i < len(initial_positions):
                self.particles.append(Particle(dimensions, initial_positions[i]))
            else:
                self.particles.append(Particle(dimensions))

        self.global_best_position = self.particles[0].position
        self.global_best_value = float('inf')
        self.iteration_best_positions = []

    def optimize(self):
        self.iteration_best_positions = []  # Reset history
        
        for _ in range(self.iterations):
            current_best_value = float('inf')
            current_best_position = None
            
            for particle in self.particles:
                value = self.objective_function(particle.position)
                if value < particle.best_value:
                    particle.best_value = value
                    particle.best_position = list(particle.position)
                if value < self.global_best_value:
                    self.global_best_value = value
                    self.global_best_position = list(particle.position)
                if value < current_best_value:
                    current_best_value = value
                    current_best_position = list(particle.position)

            # Store the best position for this iteration
            if current_best_position:
                self.iteration_best_positions.append(
                    (current_best_position, current_best_value)
                )

            for particle in self.particles:
                particle.update_velocity(self.global_best_position, 
                                      self.inertia, self.cognitive, self.social)
                particle.update_position()

        return self.global_best_position, self.global_best_value

    def get_optimization_history(self):
        """Return the history of best positions and their values"""
        return self.iteration_best_positions

# Example usage:
def objective_function(x):
    return sum([xi**2 for xi in x])

