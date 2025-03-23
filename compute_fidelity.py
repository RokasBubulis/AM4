import numpy as np
import matplotlib.pyplot as plt
from qutip import tensor, num, qeye, create, destroy, basis, sesolve, expect, mesolve # type: ignore

def compute_fidelity(omega_array, g_0_array, alpha,
                     omega_photon, omega_phonon, photon_loss_rate,
                     truncation_photon_space, truncation_phonon_space, set_g_0= None):
    
    index_photon_omega = np.abs(omega_array - omega_photon).argmin()

    # Extract the corresponding g_0 value from g_0_array
    g_0_at_photon_omega = g_0_array[index_photon_omega]

    if set_g_0 is not None:
        print(f'g_0 is set manually to: {set_g_0}')
        g_0_at_photon_omega = set_g_0

    truncation_photon_space = 10
    truncation_phonon_space = 20
    
    omega_photon_copy = omega_photon
    omega_phonon /= omega_photon
    g_0_at_photon_omega /= omega_photon
    omega_photon = 1

    tmax = 20
    tlist = np.linspace(0, tmax, 1000)

    phonon_term = omega_phonon * tensor(num(truncation_phonon_space), qeye(truncation_photon_space))
    cdagger_c_product = (alpha*qeye(truncation_photon_space)+create(truncation_photon_space)) @ (alpha*qeye(truncation_photon_space)+destroy(truncation_photon_space))
    photon_term = omega_photon * tensor(qeye(truncation_phonon_space), cdagger_c_product)

    coupling_term = g_0_at_photon_omega * tensor(create(truncation_phonon_space) + destroy(truncation_phonon_space), cdagger_c_product)

    H = phonon_term + photon_term + coupling_term
    if photon_loss_rate == 0:
        loss_operator = []
    else:
        loss_operator = [np.sqrt(photon_loss_rate) * tensor(qeye(truncation_phonon_space), destroy(truncation_photon_space))]
    initial_state = tensor(basis(truncation_phonon_space, 0), basis(truncation_photon_space, 1))
    evolved_states = mesolve(H, initial_state, tlist, loss_operator)

    # Target states
    target_state10 = tensor(basis(truncation_phonon_space, 1), basis(truncation_photon_space, 0))
    target_state20 = tensor(basis(truncation_phonon_space, 2), basis(truncation_photon_space, 0))
    target_state21 = tensor(basis(truncation_phonon_space, 2), basis(truncation_photon_space, 1))
    target_state22 = tensor(basis(truncation_phonon_space, 2), basis(truncation_photon_space, 2))
    target_state31 = tensor(basis(truncation_phonon_space, 3), basis(truncation_photon_space, 1))
    target_state42 = tensor(basis(truncation_phonon_space, 4), basis(truncation_photon_space, 2))

    fidelity_array_10 = [np.abs(target_state10.overlap(state))**2 for state in evolved_states.states]
    fidelity_array_20 = [np.abs(target_state20.overlap(state))**2 for state in evolved_states.states]
    fidelity_array_21 = [np.abs(target_state21.overlap(state))**2 for state in evolved_states.states]
    fidelity_array_31 = [np.abs(target_state31.overlap(state))**2 for state in evolved_states.states]
    fidelity_array_22 = [np.abs(target_state22.overlap(state))**2 for state in evolved_states.states]
    fidelity_array_42 = [np.abs(target_state42.overlap(state))**2 for state in evolved_states.states]

    # Create a figure with 1 row and 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot Fidelity vs Time in the first subplot
    axes[0].plot(tlist, fidelity_array_10, label=r'$\vert 10 \rangle$')
    axes[0].plot(tlist, fidelity_array_20, label=r'$\vert 20 \rangle$')
    axes[0].plot(tlist, fidelity_array_21, label=r'$\vert 21 \rangle$')
    # axes[0].plot(tlist, fidelity_array_22, label=r'$\vert 22 \rangle$')
    # axes[0].plot(tlist, fidelity_array_31, label=r'$\vert 31 \rangle$')
    # axes[0].plot(tlist, fidelity_array_42, label=r'$\vert 42 \rangle$')
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Fidelity")
    axes[0].set_title(r"Fidelity of transduction from $\vert 01 \rangle$")
    axes[0].legend()
    axes[0].grid(True)

    # Photon and Phonon Expectation vs Time in the second subplot
    photon_number_operator = tensor(qeye(truncation_phonon_space), num(truncation_photon_space))
    phonon_number_operator = tensor(num(truncation_phonon_space), qeye(truncation_photon_space))

    photon_expectation = expect(photon_number_operator, evolved_states.states)
    phonon_expectation = expect(phonon_number_operator, evolved_states.states)

    axes[1].plot(tlist, photon_expectation, label="Photons")
    axes[1].plot(tlist, phonon_expectation, label="Phonons")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("<n>")
    axes[1].legend()
    axes[1].set_title("<n> vs Time")
    axes[1].grid(True)

    # Show the plots
    fig.suptitle(rf'Input: $\lambda_c = {(2*np.pi*3*10**8/omega_photon_copy*10**6):.2f} \, \mu m$' 
                '\n'  # Correct way to break the line
                rf'$\omega_m = {omega_phonon:.2f} \, \omega_c$, $g_0 = {g_0_at_photon_omega:.2e} \, \omega_c$, $\alpha =$ {alpha}')

    plt.tight_layout()  # Adjust layout for better spacing
    plt.show()