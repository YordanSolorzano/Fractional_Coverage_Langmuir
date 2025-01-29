# Import Libraries
# Standard library
import os

# Third-party libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ASE (Atomic Simulation Environment)
from ase.units import kB  # Boltzmann constant

# Animations Library
import glob
from PIL import Image



class SurfaceCoverage:
    """
    A model to calculate the surface coverage fraction as a function of binding energy and temperature,
    using a Langmuir adsorption isotherm-based approach and a proportionality factor with the adsorption surface area.
    """
    
    def __init__(self, n_adatoms, hexagonal, rhombile, triangular, C_nPh6, T_max, T_min, n_sites):
        """
        Initialize the calculator with the necessary parameters.
        
        Parameters
        ----------
        n_adatoms : int
            Number of adatoms.
        E_bind : float
            Binding energy of the system.
        hexagonal : list of float
            Adsorption energy of the hexagonal structure.
        rhombile : list of float
            Adsorption energy of the rhombile structure.
        triangular : list of float
            Adsorption energy of the triangular structure.
        C_nPh6 : float
            Concentration of gas molecules.
        T_max : int
            Maximum temperature [K].
        T_min : int
            Minimum temperature [K].
        n_sites : int
            Number of adsorption sites (e.g., 18x24 = 432).
        """
        self.n_adatoms = n_adatoms
        self.hexagonal = hexagonal
        self.rhombile = rhombile
        self.triangular = triangular
        self.C_nPh6 = C_nPh6
        self.T_max = T_max
        self.T_min = T_min
        self.n_sites = n_sites
        
    def probabilities(self, E_bind):
        """
        A model to calculate the surface coverage fraction as a function of binding energy and temperature,
        using a Langmuir adsorption isotherm-based approach and a proportionality factor with the adsorption surface area.
        """
        E_bind_float = float(self.E_bind)
        temperature = np.linspace(self.T_min, self.T_max, self.T_max - self.T_min + 1)
        kb_T = temperature * kB
        
        isolated = 0
        theta_Auad = np.exp(E_bind_float / (kB * temperature)) / (1 + np.exp(E_bind_float / (kB * temperature)))

        data = {'Temperatura': temperature}
        data_pxc = {'Temperatura': temperature}
        K_ads = np.exp((4.311064999999871) / kb_T)
        
        p_nad_0 = (1 - theta_Auad) ** self.n_sites
        data['P_Nad_0'] = p_nad_0

        K_Ph6_0  = np.exp((isolated)   / (kB * temperature))        
        K_3Ph6_0 = np.exp(-(self.hexagonal[0]) / (kB * temperature))     
        K_6Ph6_0 = np.exp(-(self.rhombile[0])   / (kB * temperature))   
        K_9Ph6_0 = np.exp(-(self.triangular[0]) / (kB * temperature))
        
        theta_clean_0 = 1 / (1 + self.C_nPh6 * K_ads * (1 + K_3Ph6_0 + K_6Ph6_0 + K_9Ph6_0))

        theta_Ph6_0  = theta_clean_0 * self.C_nPh6 * K_ads
        theta_3Ph6_0 = theta_Ph6_0 * K_3Ph6_0
        theta_6Ph6_0 = theta_Ph6_0 * K_6Ph6_0
        theta_9Ph6_0 = theta_Ph6_0 * K_9Ph6_0
        
        data_0 = {
            "Temperatur [K]": temperature ,
            f"theta_3Ph6": p_nad_0 * theta_3Ph6_0,
            f"theta_6Ph6": p_nad_0 * theta_6Ph6_0,
            f"theta_9Ph6": p_nad_0 * theta_9Ph6_0,
            f"theta_clean": p_nad_0 * theta_clean_0,
            f"theta_isolated": p_nad_0 * theta_Ph6_0
        }
        
        probability_0 = pd.DataFrame(data_0)
        probability_0.to_csv(f'./results/coverage_adatom_0_{self.E_bind}.dat', sep=" ", index=False, header=False)
        
        coverage_0 = probability_0['theta_3Ph6'] + probability_0['theta_6Ph6'] + probability_0['theta_9Ph6'] + probability_0['theta_clean'] + probability_0['theta_isolated']

        coverage_acum = coverage_0.copy()
        coverage_acum_2 = coverage_0.copy()
        fractional_ac_coverage = {'Temperatura': temperature, 'coverage_0': coverage_0}
        
        sns.set_style("ticks")  
        plt.rcParams['font.family'] = 'serif'  
        plt.rcParams['font.serif'] = 'Times New Roman'  
        plt.rcParams['font.size'] = 12  

        plt.plot(temperature, coverage_0, color='black', linewidth=0.3)
        plt.fill_between(temperature, 0, coverage_0, color='cyan', alpha=0.8, label='No adatom')
        
        for i in range(1, self.n_adatoms + 1):
            p_nad_i = (1 - sum(data[f'P_Nad_{j}'] for j in range(i))) * (1 - theta_Auad) ** (self.n_sites - i)
            data[f'P_Nad_{i}'] = p_nad_i

            K_Ph6  = np.exp((isolated)   / (kB * temperature))
            K_3Ph6 = np.exp(-(self.hexagonal[i]) / (kB * temperature))
            K_6Ph6 = np.exp(-(self.rhombile[i])   / (kB * temperature))
            K_9Ph6 = np.exp(-(self.triangular[i]) / (kB * temperature))

            theta_clean = 1 / (1 + self.C_nPh6 * K_ads * (1 + K_3Ph6 + K_6Ph6 + K_9Ph6))

            theta_Ph6  = theta_clean * self.C_nPh6 * K_ads
            theta_3Ph6 = theta_Ph6 * K_3Ph6
            theta_6Ph6 = theta_Ph6 * K_6Ph6
            theta_9Ph6 = theta_Ph6 * K_9Ph6

            data_pxc = {
                "Temperatur [K]": temperature ,
                f"theta_3Ph6": p_nad_i * theta_3Ph6,
                f"theta_6Ph6": p_nad_i * theta_6Ph6,
                f"theta_9Ph6": p_nad_i * theta_9Ph6,
                f"theta_clean": p_nad_i * theta_clean,
                f"theta_isolated": p_nad_i * theta_Ph6
            }

            probability = pd.DataFrame(data_pxc)
            probability.to_csv(f'./results/coverage_{i}_adatom__{self.E_bind}.dat', sep=" ", index=False, header=False)

            coverage = probability['theta_3Ph6'] + probability['theta_6Ph6'] + probability['theta_9Ph6'] + probability['theta_clean'] + probability['theta_isolated']
            coverage_acum += coverage.copy()
            fractional_ac_coverage[f'coverage_{i}'] = coverage_acum.copy()
            
            plt.plot(temperature, coverage_acum.copy(), color='black', linewidth=0.3)
            plt.fill_between(temperature, coverage_acum_2, coverage_acum.copy(), alpha=0.8, label=f'{i} adatom')
            
            coverage_acum_2 += coverage.copy()

        p_nad_more = 1 - sum(data[f'P_Nad_{j}'] for j in range(self.n_adatoms + 1))
        data['P_Nad_more'] = p_nad_more

        data_pxc_more = {
            "Temperatur [K]": temperature ,
            f"theta_3Ph6": p_nad_more * theta_3Ph6,
            f"theta_6Ph6": p_nad_more * theta_6Ph6,
            f"theta_9Ph6": p_nad_more * theta_9Ph6,
            f"theta_clean": p_nad_more * theta_clean,
            f"theta_isolated": p_nad_more * theta_Ph6
        }

        probability_more = pd.DataFrame(data_pxc_more)
        probability_more.to_csv(f'./results/coverage_adatom_more_{self.E_bind}.dat', sep=" ", index=False, header=False)

        coverage_more = probability_more['theta_3Ph6'] + probability_more['theta_6Ph6'] + probability_more['theta_9Ph6'] + probability_more['theta_clean'] + probability_more['theta_isolated']
        coverage_acum += coverage_more

        plt.plot(temperature, coverage_acum, color='black', linewidth=0.3)
        plt.fill_between(temperature, coverage_acum_2, 1 , color='black', alpha=0.8, label='more adatom')
        
        fractional_ac_coverage['coverage_more'] = coverage_acum.copy()

        fractional_ac_coverage_set = pd.DataFrame(fractional_ac_coverage)
        fractional_ac_coverage_set.to_csv(f'./results/fractional_coverage_saved_{self.E_bind}.dat', sep=" ", index=False, header=False)

        plt.ylim(0, 1)
        plt.xlim(200, 700)

        legend = plt.legend(frameon=True)
        for text in legend.get_texts():
            plt.setp(text, color='black')
        frame = legend.get_frame()
        frame.set_color('white')
        frame.set_edgecolor('black')

        plt.gca().spines['top'].set_color('black')
        plt.gca().spines['right'].set_color('black')
        plt.gca().spines['bottom'].set_color('black')
        plt.gca().spines['left'].set_color('black')

        plt.title(rf'Coverage of Network Formation on Au(111) with E_bind = {self.E_bind} [eV/mol]')
        plt.xlabel('Temperatura [K]', fontsize=12, fontname='Times New Roman')
        plt.ylabel('Fractional Coverage [u.a.]', fontsize=12, fontname='Times New Roman')

        plt.savefig(f'./results/network_formation_{self.E_bind}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def coverage_fractional_E_bind(self, e_bin_min, e_bin_max):
        """
        Calculated fractional coverage for an array of energies of binding 

        Output:  None
        """
        num_points = int((abs(e_bin_max - e_bin_min) + 0.01) * 100)
        energies = np.linspace(e_bin_min, e_bin_max, num_points)
        energies_formatted = [f"{energy:.2f}" for energy in energies]

        for e_binding in energies_formatted:
            self.E_bind = e_binding
            self.probabilities(e_binding)

        images_in = glob.glob("./results/network_formation_****.png")
        images = [Image.open(img) for img in sorted(images_in)]

        gif_image_out = f"./results/animation_{self.n_adatoms}_adatoms.gif"
        images[0].save(gif_image_out, save_all=True, append_images=images[1:], duration=400, loop=0)
        
        print(f"GIF guardado como {gif_image_out}")
        return "Complete Calculations"
