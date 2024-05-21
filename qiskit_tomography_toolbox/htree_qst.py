from typing import List, Optional, Tuple
import qiskit
from qiskit import QuantumCircuit
from qiskit.primitives import Sampler
import numpy as np
import treelib
from scipy import sparse
from .base_tomography import BaseTomography


class HTreeQST(BaseTomography):
    """Tomography of real wave function based on a Hadammard Tree."""

    def __init__(
        self,
        circuit: QuantumCircuit,
        sampler: Sampler,
        use_matrix_path: Optional[bool] = True,
    ):
        """Perform a QST for real valued state vector
        This needs only N additional circuits but require some posprocesing

        Args:
            circuit (QuantumCircuit): the base circuit we want to evaluate
            sampler (Sampler): A sampler primitive
        """
        # store the roots/leaves of each level for faster processing
        self.root: List = []
        self.leaf: List = []

        # circuit and size
        self.circuit = circuit
        self.num_qubits = circuit.num_qubits
        self.size = 2**self.num_qubits

        # get the full connection tree
        self.tree = self.get_tree()

        # get all the paths
        self.path_to_node = self.get_path()
        if use_matrix_path:
            self.path_matrix = self.get_path_sparse_matrix()
            self.idx_path_matrix = np.insert(
                np.cumsum(self.path_matrix.getnnz(axis=1)), 0, 0
            )
        else:
            self.path_matrix = None
            self.idx_path_matrix = None

        # sampler and circuits
        self.sampler = sampler
        self.list_circuits = self.get_circuits()
        self.ncircuits = len(self.list_circuits)

    def get_tree(self) -> List:
        """Compute the tree"""

        def init_tree() -> Tuple[List, List, List]:
            """initialize the tree

            Returns:
                tree: tree
            """
            trees = []
            level_root, level_leaf = [], []
            for i in range(int(self.size / 2)):
                tree = treelib.Tree()
                a, b = 2 * (i), 2 * (i) + 1
                tree.create_node(a, a, data=1)
                tree.create_node(b, b, parent=a)
                trees.append(tree)

                level_leaf.append(b)
                level_root.append(a)
            return trees, level_root, level_leaf

        def link_trees(trees: List) -> Tuple[int, List, List]:
            """link multiple tress

            Args:
                trees (list): list of trees
            """
            ntree = len(trees)
            level_root, level_leaf = [], []
            for iter_ in range(1, self.num_qubits):
                root, leaf = [], []
                for iroot in range(0, int(ntree), 2**iter_):
                    new_root = trees[iroot].root
                    new_leaf = iroot + 2 ** (iter_ - 1)
                    trees[iroot].paste(new_root, trees[new_leaf])

                    root.append(trees[iroot].root)
                    leaf.append(trees[new_leaf].root)

                level_root.append(root)
                level_leaf.append(leaf)
            return trees[0], level_root, level_leaf

        tree_list, root, leaf = init_tree()
        self.root.append(root)
        self.leaf.append(leaf)
        tree, root, leaf = link_trees(tree_list)
        self.root += root
        self.leaf += leaf

        return tree  # type: ignore[return-value]

    def get_path(self) -> List:
        """Create the paths between the root and all the leaves"""
        paths = []
        for inode in range(self.size):
            paths.append(list(self.tree.rsearch(inode)))  # type: ignore[attr-defined]
        return paths

    def get_path_sparse_matrix(self) -> sparse.sparray:
        """transforms the path into a sparse matrix

        Returns:
            coo matrix: sparse matrix of the path
        """
        row_idx, col_idx, vals = [], [], []
        for ip, path in enumerate(self.path_to_node):
            num_nodes = len(path)
            row_idx += [ip] * num_nodes
            col_idx += path
            vals += [1] * num_nodes
        return sparse.coo_matrix(
            (vals, (row_idx, col_idx)), shape=(self.size, self.size)
        )

    def get_circuits(self) -> List:
        """Create the circuits containing a single H on a given qubit after the circuit

        Args:
            circuits (list): List of circuits
        """
        list_circuits = [self.circuit.measure_all(inplace=False)]

        for iq in range(self.num_qubits):
            qc = qiskit.QuantumCircuit(self.num_qubits)
            qc.append(self.circuit, range(self.num_qubits))
            qc.h(iq)
            list_circuits.append(qc.measure_all(inplace=False))
        return list_circuits

    def get_samples(self, parameters: np.ndarray) -> List:
        """Sample the circuit

        Args:
            parameters (np.array): values of the variational parameters of the circuit
        """
        results = (
            self.sampler.run(self.list_circuits, [parameters] * self.ncircuits)
            .result()
            .quasi_dists
        )
        samples = []
        for res in results:
            proba = np.zeros(2**self.num_qubits)
            for k, v in res.items():
                proba[k] = v
            samples.append(proba)
        return samples

    def get_weight(self, samples: List) -> np.ndarray:
        """Get the relative sign between parent/child node

        Args:
            samples (list): lit of samples of circuits
        """
        # root
        weights = np.zeros_like(samples[0])
        weights[0] = 1

        # link the weights
        for iter_ in range(0, self.num_qubits):
            roots = self.root[iter_]
            leafs = self.leaf[iter_]
            signs = np.sign(
                2 * samples[iter_ + 1][roots] - samples[0][roots] - samples[0][leafs]
            )
            weights[leafs] = signs

        return weights

    def get_signs(self, weights: np.ndarray) -> np.ndarray:
        """Compute the signs of each components

        Args:
            weights (np.array): relative sign between parent/child in the tree
        """

        # if the path is not known
        if self.path_matrix is None:
            signs = np.zeros_like(weights)
            for ip, path in enumerate(self.path_to_node):
                signs[ip] = weights[path].prod()
            return signs

        # otherwise use the path
        mat = self.path_matrix.multiply(weights)
        return np.multiply.reduceat(mat.data, self.idx_path_matrix[:-1])

    def get_relative_amplitude_sign(self, parameters: np.ndarray) -> np.ndarray:
        """Get the relative amplitude of each components relative to the root

        Args:
            parameters (np.array): values of the variational parameters of the circuit
        """
        samples = self.get_samples(parameters)
        weights = self.get_weight(samples)
        return self.get_signs(weights)

    def get_statevector(self, parameters: np.ndarray) -> np.ndarray:
        """Get the statevector of the circuit

        Args:
            parameters (np.array): values of the variational parameters of the circuit
        """
        samples = self.get_samples(parameters)
        if np.any(samples[0] < 0):
            print("Warning : Negative sampling values found in HTree")
            amplitudes = np.sqrt(np.abs(samples[0]))
        else:
            amplitudes = np.sqrt(samples[0])
        weights = self.get_weight(samples)
        signs = self.get_signs(weights)
        return amplitudes * signs

    def get_density_matrix(self, parameters: np.ndarray) -> np.ndarray:
        """Get the density matrix of the circuit

        Args:
            parameters (np.ndarray): parameter of the circuit

        Returns:
            np.ndarray: density matrix
        """
        vector = self.get_statevector(parameters)
        return np.outer(vector, vector)
