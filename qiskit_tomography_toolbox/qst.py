import numpy as np
from qiskit_experiments.library import StateTomography
from .base_tomography import BaseTomography


class FullQST(BaseTomography):
    def __init__(self, circuit, backend, shots=1000):
        self.backend = backend
        self.circuit = circuit
        self.shots = shots

    def get_relative_amplitude_sign(self, parameters):
        """_summary_

        Args:
            circuit (_type_): _description_
            parameters (_type_): _description_
            backend (_type_): _description_
        """

        density_matrix = self.get_density_matrix(parameters)
        return self.extract_sign(density_matrix)

    @staticmethod
    def extract_sign(density_matrix):
        """_summary_

        Args:
            density_matrix (_type_): _description_
        """
        return np.sign(density_matrix[0, :].real)

    @staticmethod
    def extract_statevector(density_matrix):
        """_summary_

        Args:
            density_matrix (_type_): _description_
        """

        signs = np.sign(density_matrix[0, :].real)
        amplitudes = np.sqrt(np.diag(density_matrix).real)
        return signs * amplitudes

    def get_density_matrix(self, parameters):
        """_summary_

        Args:
            parameters (_type_): _description_

        Returns:
            _type_: _description_
        """
        qstexp1 = StateTomography(self.circuit.assign_parameters(parameters))
        qstdata1 = qstexp1.run(self.backend, shots=self.shots).block_for_results()
        return qstdata1.analysis_results("state").value.data.real

    def get_statevector(self, parameters):
        """_summary_

        Args:
            parameters (_type_): _description_
        """
        density_matrix = self.get_density_matrix(parameters)
        return self.extract_statevector(density_matrix)
