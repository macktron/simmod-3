import random
import numpy as np
import matplotlib.pyplot as plt



##################### Metropolis Algorithm #####################
def generate_candidate( x : float, delta : float) -> float :
    """
    Generate next candidate by adding U from the previous candidate x, 
    where U is drawn from the uniform dustribution [-delta, delta]. 
    """
    return x + np.random.uniform(-delta, delta)

def p(x : float) -> float :
    """
    Our non normalized probability function.
    """
    return 0 if x < 0 else np.exp(-x)

def evaluate_candidate(x_previous : float, x_candidate : float) -> bool:
    """
    Evaluate the candidate by calculating the evaluation quota and comparing it to a random number
    between 0 and 1. If the evaluation quota is greater than the random number, accept the candidate.
    """
    evaluation_quota = p(x_candidate)/p(x_previous)

    if evaluation_quota >= 1:
        return True
    elif evaluation_quota >= random.random():
        return True
    else:
        return False
    
def metropolis_algorithm( delta : float = 0.1, Nmax : int = 1000, N0 : int = 10 ) -> list[float]:
    """
    The metropolis algorithm.
    """
    
    i = 1
    sequence = [0]

    while i < Nmax:
        x_previous = sequence[-1]
        x_candidate = generate_candidate(x_previous, delta)

        if evaluate_candidate(x_previous, x_candidate):
            sequence.append(x_candidate)
        else:
            sequence.append(x_previous)
        i += 1
    return sequence[N0:]
##################### Metropolis Algorithm #####################



##################### Plotting and Analysis #####################
def calculate_sucess_rate(sequence : list[float]) -> float:
    """
    Calculate the sucess rate of the metropolis algorithm by counting the number of duplicates
    """
    duplicate = 0
    length = len(sequence)
    for i in range(1,length):
        if sequence[i] == sequence[i-1]:
            duplicate += 1
    return 1 - duplicate/length

def get_SEM_RMS_MAR_over_delta(delta_values : list[float], N : int, runs : int) -> tuple[dict[float], dict[float], dict[float]]:
    """
    Perform runs and return the SEM, RMS and MAR results over the delta values.
    For each delta value, the algorithm is run 'runs' times and the results are averaged.
    """
    SEM_dict = {}        # Standard Error of the Mean
    RMS_dict = {}        # Root Mean Square
    MAR_dict = {}        # Metropolis Acceptance Rate
    for delta in delta_values:
        SEM = []
        RMS = []
        for _ in range(runs):
            samples = metropolis_algorithm(delta, N, 100)
            mean_sample = np.mean(samples)
            SEM.append(np.std(samples) / np.sqrt(N))
            RMS.append(np.sqrt(np.mean((mean_sample - 1)**2)))
        
        SEM_dict[delta] = np.mean(SEM)
        RMS_dict[delta] = np.mean(RMS)
    return SEM_dict, RMS_dict

def get_SEM_RMS_MAR_over_N(delta : float, N_values : list[int], runs : int) -> tuple[dict[int], dict[int], dict[int]]:
    """
    Perform runs and return the SEM, RMS and MAR results over the N values.
    For each N value, the algorithm is run 'runs' times and the results are averaged.
    """
    SEM_dict = {}        # Standard Error of the Mean
    RMS_dict = {}        # Root Mean Square
    for N in N_values:
        SEM = []
        RMS = []
        for _ in range(runs):
            samples = metropolis_algorithm(delta, N, 100)
            mean_sample = np.mean(samples)
            SEM.append(np.std(samples) / np.sqrt(N))
            RMS.append(np.sqrt(np.mean((mean_sample - 1)**2)))
        
        SEM_dict[N] = np.mean(SEM)
        RMS_dict[N] = np.mean(RMS)
    return SEM_dict, RMS_dict

def plot_SEM_RMS(SEM : dict[float], RMS : dict[float]) -> None:
    """
    Plot the SEM, RMS over the delta values.
    """
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(list(SEM.keys()), list(SEM.values()))
    plt.xscale('log')
    plt.xlabel('Delta')
    plt.ylabel('Standard Error (SEM)')
    plt.title('SEM vs Delta')

    plt.subplot(1, 2, 2)
    plt.plot(list(RMS.keys()), list(RMS.values()), color='orange')
    plt.xscale('log')
    plt.xlabel('Delta')
    plt.ylabel('RMS Difference')
    plt.title('RMS Difference vs Delta')

    plt.tight_layout()
    plt.savefig('figures/3.1.pdf')
    plt.show()

def plot_SEM_RMS_MAR_over_N(SEM_delta_N : dict[dict[int]], RMS_delta_N : dict[dict[int]]) -> None:
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    for delta, SEM in SEM_delta_N.items():
        plt.plot(list(SEM.keys()), list(SEM.values()), label=f'delta = {delta}')
    plt.xscale('log')
    plt.xlabel('N')
    plt.ylabel('Standard Error (SEM)')
    plt.title('SEM vs N')
    plt.legend()

    plt.subplot(1, 2, 2)
    for delta, RMS in RMS_delta_N.items():
        plt.plot(list(RMS.keys()), list(RMS.values()), label=f'delta = {delta}')
    plt.xscale('log')
    plt.xlabel('N')
    plt.ylabel('RMS Difference')
    plt.title('RMS Difference vs N')
    plt.legend()

    plt.tight_layout()
    plt.savefig('figures/3.1_over_N.pdf')
    plt.show()


##################### Plotting and Analysis #####################



def main1():
    """
    Plot the SEM, RMS over the delta values, for a fixed N.
    """

    delta_values = [ i for i in np.arange(0.01, 10, 0.05)] # A lot of delta values, takes a minute to run
    N = 10000   # Number of steps for each run
    runs = 20   # Number of samples for each delta value

    SEM_dict, RMS_dict = get_SEM_RMS_MAR_over_delta(delta_values, N, runs)
    plot_SEM_RMS(SEM_dict, RMS_dict)

def main2():
    """
    Plot the SEM, RMS over the N values, for a fixed delta, but a few different delta values.
    """

    delta = [0.01, 0.1, 1, 10]
    N_values = [ int(i) for i in np.linspace(10, 100000, 20)]

    runs = 10   # Number of samples for each delta value
    RMS_dict_delta_N = {}
    SEM_dict_delta_N = {}
    for delta in delta:
        print(f"delta = {delta}")
        SEM_dict, RMS_dict = get_SEM_RMS_MAR_over_N(delta, N_values, runs)
        RMS_dict_delta_N[delta] = RMS_dict
        SEM_dict_delta_N[delta] = SEM_dict
    plot_SEM_RMS_MAR_over_N(SEM_dict_delta_N, RMS_dict_delta_N)

if __name__ == "__main__":
    main2()