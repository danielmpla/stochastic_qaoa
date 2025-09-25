import numpy as np
from matplotlib import pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit.visualization import plot_histogram
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2


class AmplitudeEncoding:
    def __init__(self, probability_distribution):
        self.probability_distribution = probability_distribution

        self.highest_value = self._get_highest_value()
        self.number_of_qubits = int(np.ceil(np.log2(self.highest_value + 1)))

        self._fill_probability_distribution()


    def _get_highest_value(self):
        highest_value = 0

        for tp in self.probability_distribution:
            if tp[0] > highest_value:
                highest_value = tp[0]

        return highest_value

    def _fill_probability_distribution(self):
        value_set = {k for k, v in self.probability_distribution}
        new_probability_distribution = []

        for i in range(2 ** self.number_of_qubits):
            if i not in value_set:
                new_probability_distribution.append([i, 0])
            else:
                probability = None
                for k, v in self.probability_distribution:
                    if  k == i:
                        probability = v
                        continue

                new_probability_distribution.append([i, np.sqrt(probability)])

        self.probability_distribution = new_probability_distribution
        self.probability_distribution.sort(key=lambda x: x[0])

    def create_amplitude_encoding_circuit(self, measured=False):
        circ = QuantumCircuit(self.number_of_qubits)

        circ.initialize([v for k, v in self.probability_distribution], [i for i in range(self.number_of_qubits)], normalize=False)

        if measured:
            circ.measure_all()

        return circ


if __name__ == "__main__":
    probability_distribution = [(1, 0.3), (2, 0.5), (3, 0.2)]

    ae = AmplitudeEncoding(probability_distribution)
    circ = ae.create_amplitude_encoding_circuit(measured=True)

    circ.decompose().decompose().decompose().decompose().decompose().draw('mpl')
    plt.show()

    sim = AerSimulator()
    tqc = transpile(circ, sim, optimization_level=3)

    sampler = SamplerV2()

    result = sampler.run([tqc], shots=1000).result()
    plot_histogram(result[0].data.meas.get_counts())
    plt.show()
