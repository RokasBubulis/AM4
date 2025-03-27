import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def compute_g0(outcar_name, plot_min_wavelength, plot_max_wavelength, plot, cavity_length, semiconductor_length):
    assert plot_max_wavelength > plot_min_wavelength, "Invalid wavelength range for plotting"
    assert cavity_length > semiconductor_length, "Semiconductor must fit inside cavity"

    # Read OUTCAR once
    with open(outcar_name, "r") as OUTCAR:
        lines = OUTCAR.readlines()
    
    # Extract imaginary and real dielectric functions
    def extract_dielectric_function(lines, key):
        data = []
        found = False
        for line in lines:
            if key in line:
                found = True
                continue
            if found:
                line = line.strip()
                if not line:
                    break
                # Try to extract values as floats, skip lines that don't work
                try:
                    values = list(map(float, line.split()))
                    data.append(values)
                except ValueError:
                    continue  # Skip lines that can't be converted to floats
        return np.array(data) if data else None

    dataim = extract_dielectric_function(
        lines, "frequency dependent IMAGINARY DIELECTRIC FUNCTION"
    )
    datareal = extract_dielectric_function(
        lines, "frequency dependent      REAL DIELECTRIC FUNCTION"
    )

    if dataim is None or datareal is None:
        raise ValueError("Dielectric function data not found in OUTCAR.")

    # Extract volume and pressure
    volumes = []
    pressures = []
    
    for line in lines:
        if "volume of cell" in line:
            try:
                volumes.append(float(line.split()[4]))
            except ValueError:
                continue  
        if "external pressure" in line:
            try:
                pressures.append(float(line.split()[3]))
            except ValueError:
                continue  

    if volumes:
        most_frequent_volume = Counter(volumes).most_common(1)[0][0]
        cell_volume = most_frequent_volume * 1e-30  # Convert to cubic meters

    else:
        raise ValueError("No valid volume found in OUTCAR.")

    if len(set(pressures)) != 1:
        raise ValueError(f"Unexpected number of unique external pressures found: {set(pressures)}")
    
    pressure = pressures[0]

    # Process extracted data
    energy_array_eV = dataim[:, 0]
    img = dataim[:, 1]
    real = datareal[:, 1]

    h = 6.62607015e-34
    c = 299792458
    hbar = h / (2 * np.pi)

    refractive_index_array = np.sqrt((np.sqrt(real**2 + img**2) + real) / 2)

    np.seterr(divide='ignore')

    energy_array_J = energy_array_eV * 1.6e-19  # Convert eV to Joules
    omega_array = energy_array_J / hbar # rad per s
    wavelength_array = (h * c) / energy_array_J # m

    # Filtering range (assuming input min/max wavelengths are in µm, and we convert them to nm for comparison)
    plot_wavelength_mask = (plot_min_wavelength < wavelength_array) & (wavelength_array < plot_max_wavelength)
    plot_omega_mask = (2*np.pi*c/plot_min_wavelength > omega_array) & (omega_array > 2*np.pi*c/plot_max_wavelength)
    filtered_wavelength = wavelength_array[plot_wavelength_mask]
    filtered_omega = omega_array[plot_omega_mask]
    filtered_refractive_index = refractive_index_array[plot_wavelength_mask]

    # Compute g_0
    v_s = 4970 # m/s longitudinal acoustic phonon velocity in GaAs (along 100) (DOI: 10.1103/PhysRevB.63.224301)
    N = semiconductor_length / cell_volume**(1/3)  
    # print(f'n cells: {N:.2e}')
    m_Ga, m_As = 69.723, 74.921  # Atomic masses of Ga and As in amu
    N_Ga, N_As = 4, 4  # Number of Ga and As atoms in a unit cell
    m = (N_Ga * m_Ga + N_As * m_As) *1.66*1e-27 # Atomic mass in kg

    # prefactor = np.sqrt(hbar * semiconductor_length / (2 * np.pi * N * m * v_s))
    # print(f'prefactor {prefactor}')
    prefactor = 1e-6 # phonon mean free path in GaAs at room temperature around 1 micrometer (https://doi.org/10.1038/srep02963)

    denominator = cavity_length + semiconductor_length * (filtered_refractive_index - 1)

    g_0_array = filtered_omega * (1 - filtered_refractive_index) / denominator * prefactor

    if plot: 
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # Adjust figsize to your preference

        # Plot refractive index vs. wavelength in the first subplot
        axes[0].plot(filtered_wavelength * 1e9, filtered_refractive_index, linestyle='-')
        axes[0].set_xlabel("Wavelength (nm)")
        axes[0].set_ylabel("Refractive index [-]")
        axes[0].grid(True)

        # Plot g_0 vs. wavelength in the first subplot
        axes[1].plot(filtered_wavelength*1e9, g_0_array)
        axes[1].set_xlabel("Wavelength [nm]")
        axes[1].set_ylabel("g_0 [rad/s]")
        axes[1].grid(True)

        # Show the plots
        plt.suptitle(f"GaAs under hydrostatic {pressure} kB pressure")
        plt.tight_layout()
        plt.show()


    return filtered_omega, g_0_array
