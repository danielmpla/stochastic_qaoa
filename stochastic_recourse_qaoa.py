import numpy as np
from matplotlib import pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.visualization import plot_histogram
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2
from scipy.optimize import minimize
from qiskit_algorithms.optimizers import ADAM, COBYLA, SPSA, QNSPSA

from amplitude_encoding import AmplitudeEncoding

class StochasticRecourseQAOA:
    def __init__(self):
        self.tqc = None
        self.sim = AerSimulator()
        self.sim.set_options(max_parallel_experiments=200, max_parallel_shots=4, max_parallel_threads=16)
        self.sampler = SamplerV2()

        self.layers = 0

        self.training = []
        self.learning = []

    def clean_learning(self):
        for x in range(0, len(self.training), 2 * self.layers):
            self.learning.append(self.training[x])

    def calculate_costs(self, bitstring):
        j_bitstring = bitstring[0:2]
        sell_bitstring = bitstring[2:4]
        buy_bitstring = bitstring[4:6]
        p_bitstring = bitstring[6:]

        j = int(j_bitstring[::-1], 2)
        sell = int(sell_bitstring[::-1], 2)
        buy = int(buy_bitstring[::-1], 2)

        p = int(p_bitstring[::-1], 2)

        P1 = 5
        P2 = 5

        costs = - j * 0.25 + buy * 0.4 - 0.1 * sell + P1 * (j - buy + sell - p) ** 2 + P2 * buy * sell

        return costs

    def compute_expectation(self, counts, shots):
        expectation = sum([(counts[key] / shots) * self.calculate_costs(key) for key in counts])
        self.training.append(expectation)

        return expectation

    def parameter_shift_gradient(self, values):
        grad = np.zeros(len(values))
        shift = np.pi / 2

        for i, p in enumerate(values):
            plus = values.copy()
            plus[i] += shift
            minus = values.copy()
            minus[i] -= shift
            grad[i] = 0.5 * (
                    self.execute_circ(values)
                    - self.execute_circ(values)
            )
        return grad

    def execute_circ(self, params):
        shots = 1024
        result = self.sampler.run([(self.tqc, params)], shots=shots).result()

        counts = result[0].data.c.get_counts()

        return self.compute_expectation(counts, shots)

    def execute_circs(self, params): # TODO: maybe use this again
        shots = 1024

        num_params = 2

        pubs = [
            (self.tqc, params[i: i + num_params])
            for i in range(0, len(params), num_params)
        ]

        result = self.sampler.run(pubs, shots=shots).result()

        return [self.compute_expectation(result[i].data.c.get_counts(), shots) for i in range(int(len(params) / num_params))]

    def create_circ(self, p, measure_qd, measure_2s=True):
        gamma = ParameterVector('gamma', p)
        beta = ParameterVector('beta', p)

        if measure_qd and measure_2s:
            qc = QuantumCircuit(8, 8)
        elif not measure_qd and measure_2s:
            qc = QuantumCircuit(8, 6)
        elif measure_qd and not measure_2s:
            qc = QuantumCircuit(8, 4)
        else:
            qc = QuantumCircuit(8, 2)

        ae = AmplitudeEncoding([(1, 0.2), (2, 0.5), (3, 0.3)]) # 0 -> p_1, 1 -> p_0

        qc.append(ae.create_amplitude_encoding_circuit(False), [i for i in range(ae.number_of_qubits)])

        qc.barrier(range(8))

        qc.h([2, 3, 4, 5, 6, 7]) # 2 -> buy_0, 3 -> buy_1, 4 -> sell_0, 5 -> sell_1, 6 -> j_0, 7 -> j_1
        # with _0 is the most left bit in the bitstring

        for i in range(p):
            # 20⋅buy₀⋅buy₁
            qc.cx(2, 3)
            qc.rz(20 * gamma[i], 3)
            qc.cx(2, 3)

            # - 40⋅buy₀⋅j₀
            qc.cx(2, 6)
            qc.rz(-40 * gamma[i], 6)
            qc.cx(2, 6)

            # - 20⋅buy₀⋅j₁
            qc.cx(2, 7)
            qc.rz(-20 * gamma[i], 7)
            qc.cx(2, 7)

            # + 40⋅buy₀⋅p₀
            qc.crz(40 * gamma[i], 1, 2) # TODO: check amplitude encoding register

            # + 20⋅buy₀⋅p₁
            qc.crz(20 * gamma[i], 0, 2)

            # - 20⋅buy₀⋅sell₀
            qc.cx(2, 4)
            qc.rz(-20 * gamma[i], 4)
            qc.cx(2, 4)

            # - 10⋅buy₀⋅sell₁
            qc.cx(2, 5)
            qc.rz(-10 * gamma[i], 5)
            qc.cx(2, 5)

            # + 20.8⋅buy₀
            qc.rz(4.8 * gamma[i], 2)

            # - 20⋅buy₁⋅j₀
            qc.cx(3, 6)
            qc.rz(-20 * gamma[i], 6)
            qc.cx(3, 6)

            # - 10⋅buy₁⋅j₁
            qc.cx(3, 7)
            qc.rz(-10 * gamma[i], 7)
            qc.cx(3, 7)

            # + 20⋅buy₁⋅p₀
            qc.crz(20 * gamma[i], 1, 3)

            # + 10⋅buy₁⋅p₁
            qc.crz(10 * gamma[i], 0, 3)

            # - 10⋅buy₁⋅sell₀
            qc.cx(3, 4)
            qc.rz(-10 * gamma[i], 4)
            qc.cx(3, 4)

            # + 5⋅buy₁⋅sell₁
            qc.cx(3, 5)
            qc.rz(5 * gamma[i], 5)
            qc.cx(3, 5)

            # + 5.4⋅buy₁
            qc.rz(5.4 * gamma[i], 3)

            # + 20⋅j₀⋅j₁
            qc.cx(6, 7)
            qc.rz(20 * gamma[i], 7)
            qc.cx(6, 7)

            # - 40⋅j₀⋅p₀
            qc.crz(-40 * gamma[i], 1, 6)

            # - 20⋅j₀⋅p₁
            qc.crz(-20 * gamma[i], 0, 6)

            # + 40⋅ j₀⋅sell₀
            qc.cx(6, 4)
            qc.rz(40 * gamma[i], 4)
            qc.cx(6, 4)

            # + 20⋅j₀⋅sell₁
            qc.cx(6, 5)
            qc.rz(20 * gamma[i], 5)
            qc.cx(6, 5)

            # + 19.5⋅j₀
            qc.rz(19.5 * gamma[i], 6)

            # - 20⋅j₁⋅p₀
            qc.crz(-20 * gamma[i], 1, 7)

            # - 10⋅j₁⋅p₁
            qc.crz(-10 * gamma[i], 0, 7)

            # + 20⋅j₁⋅sell₀
            qc.cx(7, 4)
            qc.rz(20 * gamma[i], 4)
            qc.cx(7, 4)

            # + 10⋅j₁⋅sell₁
            qc.cx(7, 5)
            qc.rz(10 * gamma[i], 5)
            qc.cx(7, 5)

            # + 4.75⋅j₁
            qc.rz(4.75 * gamma[i], 7)

            # + 20⋅p₀⋅p₁
            # constant

            # - 40⋅p₀⋅sell₀
            qc.crz(-40 * gamma[i], 1, 4)

            # - 20⋅p₀⋅sell₁
            qc.crz(-20 * gamma[i], 1, 5)

            # + 20⋅p₀
            # constant

            # - 20⋅p₁⋅sell₀
            qc.crz(-20 * gamma[i], 0, 4)

            # - 10⋅p₁⋅sell₁
            qc.crz(-10 * gamma[i], 0, 5)

            # + 4.75p₁
            # constant

            # + 20⋅sell₀⋅sell₁
            qc.cx(4, 5)
            qc.rz(20 * gamma[i], 5)
            qc.cx(4, 5)

            # + 19.8⋅sell₀
            qc.rz(19.8 * gamma[i], 4)

            # + 4.9⋅sell₁
            qc.rz(4.9 * gamma[i], 5)

            qc.barrier(range(8))

            qc.rx(2 * beta[i], 2)
            qc.rx(2 * beta[i], 3)
            qc.rx(2 * beta[i], 4)
            qc.rx(2 * beta[i], 5)
            qc.rx(2 * beta[i], 6)
            qc.rx(2 * beta[i], 7)

            qc.barrier(range(8))

        if measure_qd and measure_2s:
            qc.measure([i for i in range(8)], [i for i in range(8)])
        elif not measure_qd and measure_2s:
            qc.measure([i + 2 for i in range(6)], [i for i in range(6)])
        elif measure_qd and not measure_2s:
            qc.measure([0, 1, 6, 7], [i for i in range(4)])
        else:
            qc.measure([i + 6 for i in range(2)], [i for i in range(2)])

        self.tqc = transpile(qc, self.sim, optimization_level=3)

        return qc


if __name__ == '__main__':
    layers = [1, 2, 5, 10, 50, 150, 200]

    for p in layers:
        print(f'Starting with {p} layers')
        n_runs = 50
        shots = 1024

        all_states = set()
        per_run_percentages = []

        for i in range(n_runs):

            stochast_qaoa = StochasticRecourseQAOA()
            stochast_qaoa.layers = p

            qc = stochast_qaoa.create_circ(p, measure_qd=True)

            initial_point = [2 * np.pi * ((p - z) / p) for z in range(p)] + [2 * np.pi * (z / p) for z in range(p)]  # beta than gamma
            bounds = [(0, 2 * np.pi) for _ in range(2 * p)]

            opt = COBYLA(maxiter=10000)
            res = opt.minimize(stochast_qaoa.execute_circ, initial_point, jac=stochast_qaoa.parameter_shift_gradient, bounds=bounds)

            qc = stochast_qaoa.create_circ(p, measure_qd=False, measure_2s=False)

            result = stochast_qaoa.sampler.run([(stochast_qaoa.tqc, res.x)], shots=shots).result()

            counts = result[0].data.c.get_counts()

            # Bitstrings -> Integer und Counts -> Prozent
            pct_dict = {}
            for bitstr, cnt in counts.items():
                int_state = int(bitstr[::-1], 2)
                pct = cnt / shots * 100.0
                pct_dict[int_state] = pct
                all_states.add(int_state)

            per_run_percentages.append(pct_dict)

            print("Finished iteration", i)

        sorted_states = sorted(all_states)
        boxplot_data = []
        for s in sorted_states:
            vals = []
            for run_dict in per_run_percentages:
                vals.append(run_dict.get(s, 0.0))
            boxplot_data.append(vals)

        # Plot
        plt.figure(figsize=(max(6, len(sorted_states) * 0.6), 5))
        plt.boxplot(boxplot_data, tick_labels=[str(s) for s in sorted_states], showmeans=True)
        plt.xlabel('j')
        plt.ylabel('measure probability in %')
        plt.title(f'{n_runs} runs for {p} layers')
        plt.grid(axis='y', linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.savefig(f'boxplot_{n_runs}_{p}.png')
        plt.close()
