import numpy as np
from qiskit import QuantumCircuit
from qiskit.providers import Backend
from qiskit_experiments.library import StateTomography
from .base_tomography import BaseTomography


class FullQST(BaseTomography):
    """Perform a full qst as implemented in qiskit experiments."""

    def __init__(self, circuit: QuantumCircuit, backend: Backend, shots: int = 1000):
        """Convenience class to perform qiskit experiment tomography

        Args:
            circuit (QuantumCircuit): the circuit
            backend (Backend): the bacend
            shots (int, optional): number of shots. Defaults to 1000.
        """
        self.backend = backend
        self.circuit = circuit
        self.shots = shots

    def get_relative_amplitude_sign(self, parameters: np.ndarray) -> np.ndarray:
        """Get the relativel amplitude sign.

        Args:
            parameters (np.ndarray): varioational parameters

        Returns:
            np.ndarray: signs
        """
        density_matrix = self.get_density_matrix(parameters)
        return self.extract_sign(density_matrix)

    @staticmethod
    def extract_sign(density_matrix: np.ndarray) -> np.ndarray:
        """extract the relative signs of the different amplitudes

        Args:
            density_matrix (np.ndarray): density matrix of the circuit

        Returns:
            np.ndarray: signs
        """
        return np.sign(density_matrix[0, :].real)

    @staticmethod
    def extract_statevector(density_matrix: np.ndarray) -> np.ndarray:
        """Extract the statevector

        Args:
            density_matrix (np.ndarray): density matrix of the circuit

        Returns:
            np.ndarray: state vector
        """
        signs = np.sign(density_matrix[0, :].real)
        amplitudes = np.sqrt(np.diag(density_matrix).real)
        return signs * amplitudes

    def get_density_matrix(self, parameters: np.ndarray):
        """Get the density matrix.

        Args:
            parameters (np.ndarray): varioational parameters

        Returns:
            np.ndarray: density matrix
        """
        qstexp1 = StateTomography(self.circuit.assign_parameters(parameters))
        qstdata1 = qstexp1.run(self.backend, shots=self.shots).block_for_results()
        return qstdata1.analysis_results("state").value.data.real

    def get_state_vector(self, parameters: np.ndarray) -> np.ndarray:
        """Get the statevector.

        Args:
            parameters (np.ndarray): variational parameters

        Returns:
            np.ndarray: statevector
        """
        density_matrix = self.get_density_matrix(parameters)
        return self.extract_statevector(density_matrix)
