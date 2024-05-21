from abc import ABC, abstractmethod
import numpy as np


class BaseTomography(ABC):
    """Abstract base class for the tomography."""

    @abstractmethod
    def get_state_vector(self, parameters: np.ndarray) -> np.ndarray:
        """Get the statevector of the circuit

        Args:
            parameters (np.ndarray): potential parameters of the cricuit

        Returns:
            np.ndarray: state vector
        """
        raise NotImplementedError("Implmenent a get_state_vector method")

    @abstractmethod
    def get_density_matrix(self, parameters: np.ndarray) -> np.ndarray:
        """Get the statevector of the circuit

        Args:
            parameters (np.ndarray): potential parameters of the cricuit

        Returns:
            np.ndarray: state vector
        """
        raise NotImplementedError("Implmenent a get_density_matrix method")
