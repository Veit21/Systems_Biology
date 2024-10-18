import random
import numpy as np
import matplotlib.pyplot as plt

random.seed(10)
system_size = 100
states = [-1, 1]
iterations = 1000
temp = 2
temp_array = np.linspace(0.01, 5, 150)
interaction_energy = interaction_energy_b = 1
initial_configuration = [random.choice(states) for x in range(system_size)]


# one dimensional ising chain
def calculate_energy(configuration):
    neighboring_states = np.stack([
        configuration,
        np.roll(configuration, -1)
    ])
    hamiltonian = 0
    for i, state in enumerate(configuration):
        hamiltonian -= (state * neighboring_states[1][i] + 0.5 * (state + neighboring_states[1][i]))
    return hamiltonian


# probability that a configuration is being accepted even though the energy is higher
def r(delta_energy, temperature):
    beta = 1/temperature
    return np.exp(-beta * delta_energy)


# update the configuration n-times at temperature t
def update_configuration(config, n, t):
    current_config = config.copy()
    for i in range(n):
        random_position = random.randint(0, len(config) - 1)
        next_config = current_config.copy()
        next_config[random_position] *= -1
        current_energy = calculate_energy(current_config)
        next_energy = calculate_energy(next_config)
        d_energy = next_energy - current_energy
        if d_energy <= 0:
            current_config = next_config.copy()
        elif d_energy > 0 and random.random() <= r(d_energy, t):
            current_config = next_config.copy()
        else:
            continue
    return current_config


#  averages the ising variable in the final configuration for each of the n runs
def average_ising(initial_config, n, temp_list):
    mean_ising_list = []
    for t in temp_list:
        minimal_configuration = update_configuration(initial_config, n, t)
        num_positives = minimal_configuration.count(1)
        num_negatives = minimal_configuration.count(-1)
        mean_ising = (num_positives - num_negatives) / (num_positives + num_negatives)
        mean_ising_list.append(mean_ising)
    return mean_ising_list


if __name__ == '__main__':
    minimal_config = update_configuration(initial_configuration, iterations, temp)
    average_ising_list = average_ising(initial_configuration, iterations, temp_array)

    # plot the diagrams
    fig = plt.figure(figsize=(16, 4))
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.set_title('Initial state - t=2')
    ax1.set_xlabel('i')
    ax1.set_ylabel('sigma')
    ax1.scatter(range(system_size), initial_configuration, s=8)
    ax1.axhline(y=0, color='black')

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.set_title('Final state - t=2')
    ax2.set_xlabel('i')
    ax2.set_ylabel('sigma')
    ax2.scatter(range(system_size), minimal_config, s=8)
    ax2.axhline(y=0, color='black')

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.set_title('<Sigma> vs temperature')
    ax3.set_xlabel('temperature')
    ax3.set_ylabel('<sigma>')
    ax3.scatter(temp_array, average_ising_list, s=8)
    plt.show()

    """
    When altering the temperature, one can observe that the the system converges into a equilibrium configuration at low
    temperatures. This makes sense intuitively, because at high temperatures, the probability of accepting a new config.
    with a dE > 0 is very high and the system may quickly diverge out of a possible global minimum. At low temperatures
    however, the probability of accepting a higher energetic state is low and the system rather stays in a local minimum 
    or eventually converges into a global minimum after many iterations.
    """

    """
    Commonly the average ising variable is referred to as 'order parameter' because it is able to describe the degree of
    order within a system. As an example, if there are as many cells in state -1 than there are cells in state 1, the 
    average ising variable will be close or equal to 0 which indicates that there is little to no order within this
    system. However, if (almost) all cells are at the same state, the average ising value approaches 1 which in
    turn indicates a highly ordered system. Thus, the more the average ising variable approaches 1, the more ordered is
    the system and vice versa.
    """
