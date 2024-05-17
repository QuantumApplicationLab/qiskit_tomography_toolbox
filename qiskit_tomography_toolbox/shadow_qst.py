import numpy as np
from .base_tomography import BaseTomography


class ShadowQST(BaseTomography):
    """https://github.com/ryanlevy/shadow-tutorial"""

    def __init__(self, circuit, sampler, num_shadows):
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
    def bitGateMap(qc, g, qi):
        """Map X/Y/Z string to qiskit ops"""
        if g == "X":
            qc.h(qi)
        elif g == "Y":
            qc.sdg(qi)
            qc.h(qi)
        elif g == "Z":
            pass
        else:
            raise NotImplementedError(f"Unknown gate {g}")

    @staticmethod
    def Minv(N, X):
        """inverse shadow channel"""
        return ((2**N + 1.0)) * X - np.eye(2**N)

    @staticmethod
    def rotGate(g):
        """produces gate U such that U|psi> is in Pauli basis g"""
        if g == "X":
            return 1 / np.sqrt(2) * np.array([[1.0, 1.0], [1.0, -1.0]])
        if g == "Y":
            return 1 / np.sqrt(2) * np.array([[1.0, -1.0j], [1.0, 1.0j]])
        if g == "Z":
            return np.eye(2)
        # if we haven't returned anything yet
        raise NotImplementedError(f"Unknown gate {g}")

    def get_labels(self):
        rng = np.random.default_rng(1717)
        scheme = [
            rng.choice(["X", "Y", "Z"], size=self.num_qubits)
            for _ in range(self.num_shadows)
        ]
        return np.unique(scheme, axis=0, return_counts=True)

    def get_circuits(self):
        """_summary_"""
        list_circuits = []
        for bit_string in self.labels:
            qc = self.circuit.copy()
            for i, bit in enumerate(bit_string):
                self.bitGateMap(qc, bit, i)
            list_circuits.append(qc.measure_all(inplace=False))
        return list_circuits

    def get_samples(self, parameters):
        """_summary_

        Args:
            sampler (_type_): _description_
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

    def get_shadows(self, samples, labels=None):
        """_summary_

        Args:
            samples (_type_): _description_
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
                # for bit, count in counts.items():
                mat = 1.0
                for i, bi in enumerate(bit[::-1]):
                    b = self.rotGate(pauli_string[i])[int(bi), :]
                    mat = np.kron(self.Minv(1, np.outer(b.conj(), b)), mat)
                shadows.append(mat)
                total_count.append(count)
        return shadows, total_count

    def get_density_matrix(self, samples, labels=None):
        """_summary_

        Args:
            samples (_type_): _description_

        Returns:
            _type_: _description_
        """
        shadows, counts = self.get_shadows(samples, labels=labels)
        return np.average(shadows, axis=0, weights=counts)

    def get_relative_amplitude_sign(self, parameters):
        """_summary_

        Args:
            circuit (_type_): _description_
            parameters (_type_): _description_
            backend (_type_): _description_
        """

        samples = self.get_samples(parameters)
        rho = self.get_density_matrix(samples)
        return np.sign(rho[0, :].real)

    def get_amplitudes(self, parameters):
        """_summary_

        Args:
            parameters (_type_): _description_
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

    def get_statevector(self, parameters, samples=None, labels=None):
        """_summary_

        Args:
            parameters (_type_): _description_
        """
        if samples is None:
            samples = self.get_samples(parameters)
        rho = self.get_density_matrix(samples, labels=labels)
        signs = np.sign(rho[0, :].real)
        # amplitudes = np.sqrt(np.diag(rho).real)
        amplitudes = self.get_amplitudes(parameters)
        return signs * amplitudes

    def get_observables(self, obs, parameters):
        """_summary_

        Args:
            obs (_type_): _description_
            shadows (_type_): _description_
            counts (_type_): _description_
        """
        samples = self.get_samples(parameters)
        rho = self.get_density_matrix(samples)
        return np.trace(obs @ rho)
