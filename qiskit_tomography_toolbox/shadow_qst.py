from typing import List, Tuple, Optional, Union
import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import Sampler
from .base_tomography import BaseTomography


class ShadowQST(BaseTomography):
    """https://github.com/ryanlevy/shadow-tutorial"""

    def __init__(self, circuit: QuantumCircuit, sampler: Sampler, num_shadows: int):
        """Perform a classical shadow tomography of the circuit.

        Args:
            circuit (QuantumCircuit): The circuit we want to tomograph
            sampler (Sampler): a sampler primitive
            num_shadows (int): the numberof shadows required
        """
        self.circuit = circuit
        self.num_qubits = circuit.num_qubits
        self.sampler = sampler
        self.num_shadows = num_shadows

        if num_shadows is not None:
            # get the unique pauli strings we need
            # with the number of shots per circuit
            self.labels, self.counts = self.get_labels()
            self.max_num_shots = np.max(self.counts)

            # create the circuits we need
            self.list_circuits = self.get_circuits()
            self.ncircuits = len(self.list_circuits)

    @staticmethod
    def add_pauli_gate(circuit: QuantumCircuit, pauli: str, qi: int) -> None:
        """Add required gates to map the pauyli X, Y or Z gates.

        Args:
            circuit (QuantumCircuit): The quantum circuit
            pauli (str): pauli string
            qi (int): index of the target qubit

        Raises:
            NotImplementedError: if pauli is not X, Y or Z
        """
        if pauli == "X":
            circuit.h(qi)
        elif pauli == "Y":
            circuit.sdg(qi)
            circuit.h(qi)
        elif pauli == "Z":
            pass
        else:
            raise ValueError(f"Unknown gate {pauli}")

    @staticmethod
    def get_inverse_channel(size: int, matrix: Union[np.ndarray, float]) -> np.ndarray:
        """get tje inverse shadow channel.

        Args:
            size (int): size of the problem
            matrix (np.ndarray): input matrix

        Returns:
            np.ndarray: inerse matrix
        """
        return ((2**size + 1.0)) * matrix - np.eye(2**size)

    @staticmethod
    def rotate_gate(pauli: str) -> np.ndarray:
        """Produces gate U such that U|psi> is in Pauli basis g.

        Args:
            pauli (str): Pauli letter

        Raises:
            ValueError: is pauli is not X, Y or Z
        Returns:
            np.ndarray: rotation matrix
        """

        if pauli == "X":
            return 1 / np.sqrt(2) * np.array([[1.0, 1.0], [1.0, -1.0]])
        if pauli == "Y":
            return 1 / np.sqrt(2) * np.array([[1.0, -1.0j], [1.0, 1.0j]])
        if pauli == "Z":
            return np.eye(2)
        # if we haven't returned anything yet
        raise ValueError(f"Unknown gate {pauli}")

    def get_labels(self) -> Tuple:
        """Get a random series of label.

        Returns:
            np.ndarray: series of X, Y Z labels
        """
        rng = np.random.default_rng(1717)
        scheme = [
            rng.choice(["X", "Y", "Z"], size=self.num_qubits)
            for _ in range(self.num_shadows)
        ]
        return np.unique(scheme, axis=0, return_counts=True)

    def get_circuits(self) -> List[QuantumCircuit]:
        """Compute the circuits associated with the labels."""
        list_circuits = []
        for bit_string in self.labels:
            qc = self.circuit.copy()
            for i, bit in enumerate(bit_string):
                self.add_pauli_gate(qc, bit, i)
            list_circuits.append(qc.measure_all(inplace=False))
        return list_circuits

    def get_samples(self, parameters: np.ndarray) -> List:
        """get the samples from the circuits.

        Args:
            parameters (np.ndarray): variational parameters of the ciruits

        Returns:
            List: samples
        """
        samples = []
        num_shots = self.num_shadows
        for qc, _ in zip(self.list_circuits, self.counts):
            spl = (
                self.sampler.run(qc, parameters, shots=num_shots).result().quasi_dists
            )  # WARNING
            res = spl[0]
            proba = {}
            for k, v in res.items():
                key = np.binary_repr(k, width=self.num_qubits)
                val = int(num_shots * v)
                proba[key] = val
            samples.append(proba)
        return samples

    def get_shadows(
        self, samples: list, labels: Optional[Union[List, None]] = None
    ) -> Tuple[List[float], List[int]]:
        """Get the shadow values.

        Args:
            samples (list): samples obtained for the circuits
            labels (list, optional): Lablels of the random measurements. Defaults to None.

        Returns:
            Tuple[List, int]: values of the shadows and total count
        """
        shadows = []
        total_count = []
        if labels is None:
            labels = self.labels

        for pauli_string, counts in zip(labels, samples):
            # iterate over measurements
            for isample in range(len(counts)):
                bit = np.binary_repr(isample, width=self.num_qubits)
                if bit not in counts:
                    count = 0
                else:
                    count = counts[bit]

                mat = 1.0
                for i, bi in enumerate(bit[::-1]):
                    b = self.rotate_gate(pauli_string[i])[int(bi), :]
                    mat = np.kron(  # type: ignore[assignment]
                        self.get_inverse_channel(1, np.outer(b.conj(), b)),
                        mat,
                    )
                shadows.append(mat)
                total_count.append(count)
        return shadows, total_count

    def get_density_matrix(  # type: ignore[override] # pylint: disable=arguments-renamed
        self,
        samples: List,
        labels: Optional[Union[List, None]] = None,
    ) -> np.ndarray:
        """Get the density matrix of the ciruits.

        Args:
            samples (List): samples values
            labels (list, optional): Lablels of the random measurements. Defaults to None.

        Returns:
            np.ndarray: denity matrix
        """
        shadows, counts = self.get_shadows(samples, labels=labels)
        return np.average(shadows, axis=0, weights=counts)

    def get_relative_amplitude_sign(self, parameters: np.ndarray) -> np.ndarray:
        """get the relative amplitude signs

        Args:
            parameters (np.ndarray): variational parameters

        Returns:
            np.ndarray: relative signs
        """

        samples = self.get_samples(parameters)
        rho = self.get_density_matrix(samples)
        return np.sign(rho[0, :].real)

    def get_amplitudes(self, parameters: np.ndarray) -> np.ndarray:
        """get the  amplitude.

        Args:
            parameters (np.ndarray): variational parameters

        Returns:
            np.ndarray: amplitudes
        """
        circuit = self.circuit.measure_all(inplace=False)
        results = self.sampler.run([circuit], [parameters]).result().quasi_dists
        samples = []
        for res in results:
            proba = np.zeros(2**self.num_qubits)
            for k, v in res.items():
                proba[k] = v
            samples.append(proba)
        return np.sqrt(samples[0])

    def get_statevector(
        self,
        parameters: np.ndarray,
        samples: Optional[Union[List, None]] = None,
        labels: Optional[Union[List, None]] = None,
    ) -> np.ndarray:
        """get the statevector.

        Args:
            parameters (np.ndarray): variational parameters of th circuits
            samples (Optional[Union[List, None]], optional): sample values. Defaults to None.
            labels (Optional[Union[List, None]], optional): labels. Defaults to None.

        Returns:
            np.ndarray: state vector
        """

        if samples is None:
            samples = self.get_samples(parameters)
        rho = self.get_density_matrix(samples, labels=labels)
        signs = np.sign(rho[0, :].real)
        # amplitudes = np.sqrt(np.diag(rho).real)
        amplitudes = self.get_amplitudes(parameters)
        return signs * amplitudes

    def get_observables(self, observable: np.ndarray, parameters: np.ndarray) -> float:
        """compute the value of an observable

        Args:
            observable (np.ndarray): matrix of the observable
            parameters (np.ndarray): variational parameters of the circuits

        Returns:
            float: observable value
        """
        samples = self.get_samples(parameters)
        rho = self.get_density_matrix(samples)
        return np.trace(observable @ rho)
