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


def probabilities(n_adatoms, E_bind, hexagonal, rhombile, triangular, 
                  C_nPh6, T_max, T_min, n_sites):
 
    """
    Calculate the surface coverage when an adatom is inserted.
    
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
    
    Returns
    -------
    None
        Saves multiple CSV and PNG files with calculated data.
    """

    
    
    E_bind_float = float(E_bind)
    # Define a array Temperature 
    temperature = np.linspace(T_min, T_max, T_max - T_min + 1)
    

    kb_T = temperature*kB  # Energy [eV] equivalent to 100ºC , 25.3
    
    # Define the energy of isolated surface
    isolated = 0
    
    # Calculate theta_Auad for array Temperature
    theta_Auad = np.exp(E_bind_float / (kB * temperature)) / (1 + np.exp(E_bind_float / (kB * temperature)))

    # Define a dicctionary initial with the columna Temperature to save probabilities
    data = {'Temperatura': temperature}

    # Define a dicctionary initia to save values of probabilities*coverage
    data_pxc = {'Temperatura': temperature}

    # Define the Absorption energy &  Kinetics Adsortion
    K_ads = np.exp((4.311064999999871)/kb_T) 
    
    
    
    ######## Calculations for surface without adatoms ######
    
    # Initialization of Probability with no adatom (adatom 0) 
    p_nad_0 = (1 - theta_Auad) ** n_sites

    # Add the first column to diccitonary of probabilities 'data' which is adatom 0
    data['P_Nad_0'] = p_nad_0

    #  Arrhenius relation of kinetics 
    K_Ph6_0  = np.exp( (isolated)   /(kB * temperature))        # Isolated molecule
    K_3Ph6_0 = np.exp(-(hexagonal[0] ) /(kB * temperature))     # Hexagonal
    K_6Ph6_0 = np.exp(-(rhombile[0]  )   /(kB * temperature))   # Rhombile
    K_9Ph6_0 = np.exp(-(triangular[0]) /(kB * temperature))     # Trinagular
    
    # Calculate Thetha for 0 adatoms
    theta_clean_0 = 1 /(1 +   C_nPh6 * K_ads * (1 + K_3Ph6_0 + K_6Ph6_0 + K_9Ph6_0 ))


    #Calculate the coverage of each structure in the surface  
    theta_Ph6_0  = theta_clean_0 * C_nPh6   *  K_ads
    theta_3Ph6_0 = theta_Ph6_0   * K_3Ph6_0
    theta_6Ph6_0 = theta_Ph6_0   * K_6Ph6_0
    theta_9Ph6_0 = theta_Ph6_0   * K_9Ph6_0
    
    # Define a dicctionary to save the probabilities*coverage of non adatoms juntionsç
    
    data_0 = {
        "Temperatur [K]": temperature ,
        f"theta_3Ph6":     p_nad_0  *  theta_3Ph6_0,
        f"theta_6Ph6":     p_nad_0  *  theta_6Ph6_0,
        f"theta_9Ph6":     p_nad_0  *  theta_9Ph6_0,
        f"theta_clean":    p_nad_0  *  theta_clean_0,
        f"theta_isolated": p_nad_0  *  theta_Ph6_0
    }


    # Create the data frame 
    probability_0 = pd.DataFrame(data_0)
    
    #Save data frame of coverage with non adatoms
    probability_0.to_csv(f'./results/coverage_adatom_0_{E_bind}.dat', sep= " ", index=False,header=False)


    # fractional coverage which the sum of all networks structure
    coverage_0 = probability_0['theta_3Ph6'] + probability_0['theta_6Ph6'] + probability_0['theta_9Ph6'] + probability_0['theta_clean'] + probability_0['theta_isolated']

    ########### Finish calculations with no adatoms   ##############
    
    ###### Initializations of calculations of coverages from 1 to n adatoms  #######
    
    
    # Define the initialization variable with coverage 0 
    coverage_acum = coverage_0.copy()  # sum acummulative into the for loop
    coverage_acum_2 = coverage_0.copy()  # sum acummulative into the for loop
    # Define a dicctionary for fractional coverage acummulative 
    fractional_ac_coverage = {'Temperatura': temperature,
                          'coverage_0':  coverage_0}
    
    
    
    #Plotting 
    # Configuración estética sin cuadrícula de fondo y con bordes de ticks
    sns.set_style("ticks")  
    # Forzar uso de Times New Roman
    plt.rcParams['font.family'] = 'serif'  # Usa 'serif' como alternativa en caso de que Times New Roman no esté disponible
    plt.rcParams['font.serif'] = 'Times New Roman'  # Esto forzará a usar Times New Roman si está instalado
    plt.rcParams['font.size'] = 12  # Tamaño de fuente 12
    # Graficar las líneas
    plt.plot(temperature, coverage_0, color='black', linewidth=0.3)
    # Rellenar el área entre la curva más baja y el eje inferior (y=0)
    plt.fill_between(temperature, 0, coverage_0, color='cyan', alpha=0.8, label='No adatom')
    
    
    # Loop for to adding values into dicctionaries 
    for i in range(1, n_adatoms + 1):
        
        # Calculate using a recursive functions the probabilities of n adatoms
        p_nad_i = (1 - sum(data['P_Nad_' + str(j)] for j in range(i))) * (1 - theta_Auad) ** (n_sites - i)
        
        # Save the  values of probabilities into a dicctionary
        data['P_Nad_' + str(i)] = p_nad_i

        #Arrhenius relation according the energies i of the array defined
    
        K_Ph6  = np.exp( (isolated)   /(kB * temperature))     # Isolated molecule
        K_3Ph6 = np.exp(-(hexagonal[i] ) /(kB * temperature))     # Hexagonal
        K_6Ph6 = np.exp(-(rhombile[i]  )   /(kB * temperature))  # Rhombile
        K_9Ph6 = np.exp(-(triangular[i]) /(kB * temperature))  # Trinagular


        # Calculate Thetha 
        theta_clean = 1 /(1 +   C_nPh6 * K_ads * (1 + K_3Ph6 + K_6Ph6 + K_9Ph6 ))


        # Calculate the coverage of each structure in the surface 
        theta_Ph6  = theta_clean * C_nPh6   *  K_ads
        theta_3Ph6 = theta_Ph6   * K_3Ph6
        theta_6Ph6 = theta_Ph6   * K_6Ph6
        theta_9Ph6 = theta_Ph6   * K_9Ph6

        # Define a dicctionary to save the values of probabliites* coverage of n adatoms
        data_pxc = {
            "Temperatur [K]": temperature ,
            f"theta_3Ph6":     p_nad_i  *  theta_3Ph6,
            f"theta_6Ph6":     p_nad_i  *  theta_6Ph6,
            f"theta_9Ph6":     p_nad_i  *  theta_9Ph6,
            f"theta_clean":    p_nad_i  *  theta_clean,
            f"theta_isolated": p_nad_i  *  theta_Ph6}


        # Created the data frame of the coverage*probabilities of n adatoms
        probability = pd.DataFrame(data_pxc)

        #Save data frame 
        probability.to_csv(f'./results/coverage_{i}_adatom__{E_bind}.dat', sep= " ", index=False,header=False)

        # Define the coverage of i adatoms which is the sum total of the networks structural
        coverage = probability['theta_3Ph6'] + probability['theta_6Ph6'] + probability['theta_9Ph6'] + probability['theta_clean'] + probability['theta_isolated']

        # Sumar cobertura actual a la acumulada
        coverage_acum += coverage.copy()
        
        # Sum accumalative of n adatoms coverage*probabilities
        fractional_ac_coverage['coverage_' + str(i)] = coverage_acum.copy()
        
        # Graficar las líneas
        plt.plot(temperature, coverage_acum.copy(), color='black', linewidth=0.3)
        
        # Rellenar el área entre las curvas con diferentes colores
        plt.fill_between(temperature,coverage_acum_2, coverage_acum.copy(), alpha=0.8, label=f'{i} adatom')
        
        coverage_acum_2+= coverage.copy()
        
    ###### Calculations the other posible structures which are unstable #######
    # Calcular p_nad_more which is the non defined strucutres 
    p_nad_more = 1 - sum(data['P_Nad_' + str(j)] for j in range(n_adatoms + 1))
    
    # Save the values of p_nad_more on dicctionary of probabilities*coverage
    data['P_Nad_more'] = p_nad_more

    # Define a new dictionary of more n adatoms
    
    data_pxc_more = {
        "Temperatur [K]": temperature ,
        f"theta_3Ph6":     p_nad_more  *  theta_3Ph6,
        f"theta_6Ph6":     p_nad_more  *  theta_6Ph6,
        f"theta_9Ph6":     p_nad_more  *  theta_9Ph6,
        f"theta_clean":    p_nad_more  *  theta_clean,
        f"theta_isolated": p_nad_more  *  theta_Ph6}



    # Created the data frame 
    probability_more = pd.DataFrame(data_pxc_more)
    
    #Save data frame 
    probability_more.to_csv(f'./results/coverage_adatom_more_{E_bind}.dat', sep= " ", index=False,header=False)
    
    # define a sum of all structure difined 
    coverage_more = probability_more['theta_3Ph6'] + probability_more['theta_6Ph6'] + probability_more['theta_9Ph6'] + probability_more['theta_clean'] + probability_more['theta_isolated']

    # Update the accumulative variable to sum coverge more
    coverage_acum += coverage_more
    
    plt.plot(temperature, coverage_acum, color='black', linewidth=0.3)
    plt.fill_between(temperature, coverage_acum_2, 1 , color='black', alpha=0.8,  label='more adatom')
 
    # Add a new colummn  into the dicctionary correspont to more n adatoms
    fractional_ac_coverage['coverage_more'] = coverage_acum.copy()

    # Created the data frame 
    fractional_ac_coverage_set = pd.DataFrame(fractional_ac_coverage)

    #Save data frame 
    fractional_ac_coverage_set.to_csv(f'./results/fractional_coverage_saved_{E_bind}.dat', sep= " ", index=False,header=False)

    # Ajustar límites para asegurar visibilidad completa
    plt.ylim(0, 1)
    plt.xlim(200, 700)

    # Mejorar la leyenda
    legend = plt.legend(frameon=True)
    for text in legend.get_texts():
        plt.setp(text, color='black')  # Color de texto en negro para mejor visibilidad
    # Borde negro en la caja de leyenda
    frame = legend.get_frame()
    frame.set_color('white')  # Color de fondo blanco
    frame.set_edgecolor('black')  # Color del borde de la caja de leyenda

    # Bordes negros en el gráfico
    plt.gca().spines['top'].set_color('black')
    plt.gca().spines['right'].set_color('black')
    plt.gca().spines['bottom'].set_color('black')
    plt.gca().spines['left'].set_color('black')
    
    
    # Títulos y etiquetas de los ejes en Times New Roman
    plt.title(rf'Coverage of Network Formation on Au(111) with E_bind = {E_bind} [eV/mol]')
    plt.xlabel('Temperatura [K]', fontsize=12, fontname='Times New Roman')
    plt.ylabel('Fractional Coverage [u.a.]', fontsize=12, fontname='Times New Roman')

    plt.savefig(f'./results/network_formation_{E_bind}.png', dpi=300, bbox_inches='tight')
    # Mostrar gráfico
    #plt.show()
    plt.close()
    
    return 


def coverage_fractional_E_bind(n_adatoms, hexagonal, rhombile, triangular, C_nPh6, T_max, T_min, n_sites, e_bin_min, e_bin_max):

   
    """
    Calculate the surface coverage when n adatom is inserted and over a surface with differents energy binding 
    
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
    e_bin_min: int
        Minimun value of energy binding to create an array
    e_bin_max: int
        Maximun value of energy binding to create an array
    Returns
    -------
    None
        Saves multiple CSV, PNG files with calculated data, and 
        Create a GIF animation for a array of Energies binding
    """


    
    # Ensure it's an integer
    num_points = int((abs(e_bin_max - e_bin_min) + 0.01) * 100)
    
    # Create an array for binding energies
    energies = np.linspace(e_bin_min, e_bin_max, num_points)

    # Give a format of string  to created a geit animation 
    energies_formatted = [f"{energy:.2f}" for energy in energies]


    # Loop for each e_binding in energies_formated 
    for e_binding in energies_formatted:
        probabilities(n_adatoms,
                        e_binding, 
                        hexagonal,
                        rhombile, 
                        triangular,
                        C_nPh6,
                        T_max,
                        T_min,
                        n_sites)
    
    # Ruta a las imágenes
    images_in = glob.glob("./results/network_formation_****.png")
    
    # Cargar las imágenes
    images = [Image.open(img) for img in sorted(images_in)]
    
    # Crear el GIF
    gif_image_out = f"./results/animation_{n_adatoms}_adatoms.gif"
    images[0].save(gif_image_out, save_all=True, append_images=images[1:], duration=400, loop=0)
    
    print(f"GIF guardado como {gif_image_out}")

    
    return 


