import random
import numpy as np
import h5py


class Mutual(object):

    @staticmethod
    def get_states(num_of_spins) -> dict:

        """
            get_states() returns hashmap of integers from 0 to num_of_states
            and binary strings, corresponding to them. In other words, this
            method allows to obtain all possible states which might be observed
            in chain-like 1/2-spin magnet system.
        """

        num_of_states = 2**num_of_spins
        states = {}

        # generate states
        for i in range(0, num_of_states):
            b_state = bin(i)[2:].zfill(num_of_spins)

            states[i] = b_state

        return states


class Chain(object):

    @staticmethod
    def unsymmetrizer(symm_vec, groups, num_of_spins) -> np.ndarray:

        """
            unsymmetrizer() function converts eigenvector built in symmetrized basis
            via projectors to and eigenvector in default stationary states basis.

            We set permutation sector equal to zero, so all symmetry characters are 1.
        """

        sym_size = len(symm_vec)
        keys = list(groups.keys())
        unsymm_vec = np.zeros(2**num_of_spins)

        # i goes through symm_vec elements and keys elements which are representatives
        for i in range(0, sym_size):
            representative = keys[i]
            group = groups[representative]
            group_size = len(group)

            value = symm_vec[i] / np.sqrt(group_size)

            # j goes through states which might be obtained with representative
            for j in range(0, group_size):
                pos = group[j]
                unsymm_vec[pos] = value

        return unsymm_vec

    @staticmethod
    def build_symmetry_groups(num_of_spins, states, translation=True, parity=False) -> dict:

        """
            This function is used for building translation and parity symmetry groups
            a.k.a orbits for systems with geometry of chain.

            Full symmetry group is an outer product of translation and parity symmetry groups.

            TODO implement building symmetry groups on fly
        """

        num_of_states = 2**num_of_spins

        # build translation symmetry groups
        group_number = 1
        checked = set()
        groups = {}

        for i in range(0, num_of_states):

            if i in checked: continue

            s_state = states[i]

            checked.add(i)
            groups[group_number] = [i]

            if translation:
                for t in range(1, num_of_spins):

                    rolled_state = s_state[t:] + s_state[:t]
                    if rolled_state == s_state: break

                    key = (int(rolled_state, 2))

                    groups[group_number].append(key)
                    checked.add(key)

            if parity:
                for t in range(0, num_of_spins):

                    rolled_state = s_state[t:] + s_state[:t]
                    flipped_state = rolled_state[::-1]

                    key = (int(flipped_state, 2))

                    if key not in checked:
                        groups[group_number].append(key)
                        checked.add(key)

            group_number += 1

        # it is convenient to use representatives as dictionary keys
        new_groups = {}

        for old_key in range(1, group_number):
            new_key = min(groups[old_key])
            new_groups[new_key] = groups.pop(old_key)

        return new_groups

    @staticmethod
    def unsymm_h5_sample(h5_filename, result_filename, num_of_spins):

        """
            This function sample binary states using statevector obtained from ED
            previously converted to default basis via unsymmetrizer() function.

            As magnet system`s number of states grows exponentially, this method
            becomes inapplicable. So the main purpose of this function is debug.

            TODO delete it later
        """

        # import exact-diagonalization data from h5 file
        data_file = h5py.File(h5_filename, 'r')
        ed_vector = list(data_file['/hamiltonian/eigenvectors'])[0]

        print("Symmetrized eigenvector from ED: ", end="\n")
        print(ed_vector, "norm =", np.linalg.norm(ed_vector), end="\n\n")

        states = Mutual.get_states(num_of_spins)
        groups = Chain.build_symmetry_groups(num_of_spins, states, translation=True, parity=False)
        unsymm_vector = Chain.unsymmetrizer(ed_vector, groups, num_of_spins)

        print("Unsymmetrized eigenvector: ", end="\n")
        print(unsymm_vector, "norm =", np.linalg.norm(unsymm_vector), end="\n\n")

        shots = 8192
        weights = [amplitude ** 2 for amplitude in ed_vector]
        measurements = random.choices(range(0, len(ed_vector)), weights=weights, k=shots)

        print("Measurements: ", end="\n")
        print(measurements)

        data_file = open(result_filename, mode="w")
        for outcome in measurements:
            data_file.write(outcome+';')
        data_file.close()

    @staticmethod
    def bare_h5_sample(h5_filename, result_filename):

        """
            This function sample binary states using statevector obtained from ED
            WITHOUT converting it to default basis.

            As symmetrized basis have smaller dimension in comparsion with default,
            sample_memory() function returns specific index in symmetrized ED vector,
            instead of basis state.

            Further, this index might be randomly transformed to one of the basis states,
            which forms projector. Random choice is applicable, because all basis states
            in projector, have equal amplitudes.

            TODO integrate access to control number of shots
        """

        # import exact-diagonalization data from h5 file
        data_file = h5py.File(h5_filename, 'r')
        ed_vector = list(data_file['/hamiltonian/eigenvectors'])[0]

        print("Symmetrized eigenvector from ED: ", end="\n")
        print(ed_vector, "norm =", np.linalg.norm(ed_vector), end="\n\n")

        shots = 8192
        weights = [amplitude**2 for amplitude in ed_vector]
        measurements = random.choices(range(0, len(ed_vector)), weights=weights, k=shots)

        print("Measurements: ", end="\n")
        print(measurements)

        data_file = open(result_filename, mode="w")
        for outcome in measurements:
            data_file.write(str(outcome)+';')
        data_file.close()

    @staticmethod
    def bare_to_binary(bare_result_filename, binary_result_filename, num_of_spins):

        states = Mutual.get_states(num_of_spins)
        groups = Chain.build_symmetry_groups(num_of_spins, states, translation=False, parity=False)
        representatives = list(groups.keys())

        bare_file = open(bare_result_filename, mode="r")
        bare_samples = bare_file.read().split(';')
        bare_samples.pop(-1)
        bare_file.close()

        bare_samples = list(map(int, bare_samples))

        binary_file = open(binary_result_filename, mode="w")
        for sample in bare_samples:
            group_number = representatives[sample]
            state_index = random.choice(groups[group_number])
            binary_file.write(states[state_index])
        binary_file.close()


Chain.bare_h5_sample('h5/chain_16_ns.h5', 'bare_samples/b16_ns.dat')
Chain.bare_to_binary('bare_samples/b16_ns.dat', 'binary_samples/16_ns.dat', 16)
