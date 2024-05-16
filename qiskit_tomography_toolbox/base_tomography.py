import abc
import numpy as np


class BaseTomography(abc.ABC):
    """Abstract base class for the tomography."""

    @abc.abstractmethod
    def get_state_vector(self, parameters: np.ndarray) -> np.ndarray:
        """Get the statevector of the circuit

        Args:
            parameters (np.ndarray): potential parameters of the cricuit

        Returns:
            np.ndarray: state vector
        """
        raise NotImplementedError("Implmenent a get_state_vector method")

    @abc.abstract
    def get_density_matrix(self, parameters: np.ndarray) -> np.ndarray:
        """Get the statevector of the circuit

        Args:
            parameters (np.ndarray): potential parameters of the cricuit

        Returns:
            np.ndarray: state vector
        """
        raise NotImplementedError("Implmenent a get_density_matrix method")
