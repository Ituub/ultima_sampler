import random

import numpy as np
import qiskit
from qiskit import QuantumCircuit
import qiskit_aer
import h5py


f"""
    Mutual() class contains methods, applicable for a systems with any
    type of symmetry. For the most part it contains some service functions. (???)
"""

class Mutual(object):

    @staticmethod
    def hamming_dist(string1: str, string2: str, length: int) -> int:
        
        distance = 0

        for i in range(0, length):
            distance += abs(string1[i] - string2[i])

        return distance
    
    @staticmethod
    def specific_representative(orbit: list) -> str:

        """
            This function is used to re-define representative state
            of the orbit using specific condition a.k.a min/max int,
            structural complexity etc.
        """

        metric = 1e10
        representative_index = 0

        for i, state in enumerate(orbit):
            if int(state) < metric:
                metric = int(state)
                representative_index = i

        return orbit[representative_index]

    @staticmethod
    def get_random_basis(shots):

        th = np.ndarray((shots,), dtype=float)
        ph = np.ndarray((shots,), dtype=float)
        la = np.ndarray((shots,), dtype=float)

        total = 0
        while total != shots:
            t, p, l = np.random.random((3,))
            t = np.arccos(2 * t - 1)
            p = 2 * np.pi * p
            l = 2 * np.pi * l
            if (t <= np.pi / 2 and t >= 0 and p <= np.pi / 2 and p >= 0 and l <= np.pi / 2 and l >= 0):
                th[total] = t
                ph[total] = p
                la[total] = l
                total += 1

        return th, ph, la

    @staticmethod
    def sample_cut(choice: str, num_of_spins: int) -> str:

        """
            TODO: optimize or delete later
        """

        new_choice = ""
        ones = 0
        zeros = 0

        indexes = random.sample(range(0, num_of_spins), 4)

        # first two cycles is used to cut equal number of 0's and 1's
        """indexes = []
        while zeros != 2:
            pos = np.random.randint(0, num_of_spins // 2)
            if choice[pos] == '0' and pos not in indexes:
                indexes.append(pos)
                zeros += 1
            pos += 1

        while ones != 2:
            pos = np.random.randint(0, num_of_spins // 2)
            if choice[pos] == '1' and pos not in indexes:
                indexes.append(pos)
                ones += 1
            pos += 1"""

        for i in range(0, num_of_spins):
            if i in indexes: continue
            new_choice += choice[i]

        return new_choice

    @staticmethod
    def get_states(num_of_spins) -> dict:
        
        """
            get_states() returns hashmap of integers from 0 to num_of_states
            and binary strings, corresponding to them. In other words, this
            method allows to obtain all possible states which might be observed
            in 1/2-spin magnet system.
        """

        num_of_states = 2 ** num_of_spins
        states = {}

        # generate states
        for i in range(0, num_of_states):
            b_state = bin(i)[2:].zfill(num_of_spins)

            states[i] = b_state

        return states

    @staticmethod
    def unsymm_h5_sample(h5_filename, result_filename, num_of_spins, geometry_class):

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
        groups = geometry_class.prepare_orbits(num_of_spins, states, translation=True, parity=False)
        unsymm_vector = geometry_class.unsymmetrizer(ed_vector, groups, num_of_spins)

        print("Unsymmetrized eigenvector: ", end="\n")
        print(unsymm_vector, "norm =", np.linalg.norm(unsymm_vector), end="\n\n")

        shots = 8192
        weights = [amplitude ** 2 for amplitude in ed_vector]
        measurements = random.choices(range(0, len(ed_vector)), weights=weights, k=shots)

        print("Measurements: ", end="\n")
        print(measurements)

        data_file = open(result_filename, mode="w")
        for outcome in measurements:
            data_file.write(outcome + ';')
        data_file.close()

    @staticmethod
    def bare_h5_sample(h5_filename, result_filename) -> None:

        """
            This function sample binary states using statevector obtained from ED
            WITHOUT converting it to default basis.

            As symmetrized basis have smaller dimension in comparsion to default,
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
        representatives = list(data_file['/basis/representatives'])

        shots = 2**17
        weights = [amplitude ** 2 for amplitude in ed_vector]
        measurements = random.choices(representatives, weights=weights, k=shots)

        data_file = open(result_filename, mode="w")
        for outcome in measurements:
            data_file.write(str(outcome) + ';')
        data_file.close()

        return
    
    
    @staticmethod
    def direct_bare_to_binary(bare_filename: str, binary_filename: str, qbits: int, save_orbits=False) -> None:

        """
            This function is used to convert bare samples into binary strings,
            without taking system's symmetry into account. 

            It basicy converts integers from file with bare samples
            into binary strings, filling it with zeros to <qbits> 
        """

        bare_file = open(bare_filename, mode="r")
        bare_samples = bare_file.read().split(';')
        bare_samples.pop(-1)
        bare_file.close()

        # этого тоже касается
        checked = set()
        # orbit_file = open(binary_filename[:-4]+'+orb.dat', mode="w")
        bitstrings_file = open(binary_filename[:-4]+'.dat', mode="w")

        bare_samples = list(map(int, bare_samples))

        # binary_file = open(binary_filename, mode="w")

        for sample in bare_samples:
            state = bin(sample)[2:].zfill(qbits)
            # binary_file.write(state)

            # этот кусок надо убрать или переделать в будущем
            # тут был ---> if save_orbits <---
            if state not in checked:
                # orbit = Square.orbit_on_fly(state, 6)
                # print(''.join(orbit), file=orbit_file, end='f\n')
                print(state, file=bitstrings_file, end='\n')
            checked.add(state)

        return


    @staticmethod
    def sample_via_qiskit(qbits: int, ed_vector: np.ndarray, shots=8192) -> list:

        """
            This function is used to perform quantum state's sampling using
            qiskit.

            As bare_sample() provides the same result, which might be shown,
            main purpose of this function is debug

            TODO : delete later
        """

        grid = range(0, qbits)

        qc = qiskit.QuantumCircuit(qbits, qbits)
        qc.initialize(ed_vector, grid)

        # delete hadamar gate!!!
        # qc.sdg(grid)
        # qc.h(grid)

        qc.measure(grid, grid)

        sim = qiskit_aer.AerSimulator()
        result = sim.run(qc, shots=shots, memory=True).result()
        memory = result.get_memory()

        return memory

    @staticmethod
    def random_sample_via_qiskit(qbits: int, ed_vector: np.ndarray, shots=8192) -> list:

        """
            As the function above, this function is used to perform sampling
            using qiskit. However, this function performs sampling in random
            basis. 
            
            Currenty, bare sampling in random basis is not implemented
            in ultima_sampler

            TODO : implement random-basis-bare_sample() 
        """

        memory = []
        rh, th, ld = Mutual.get_random_basis(shots)
        sim = qiskit_aer.AerSimulator()
        grid = range(0, qbits)

        qc_init = QuantumCircuit(qbits, qbits)
        qc_init.initialize(ed_vector, grid)

        for i in range(0, shots):
            qc = qc_init.copy()

            for q in range(0, qbits):
                qc.u(rh[i], th[i], ld[i], q)

            qc.measure(grid, grid)

            result = sim.run(qc, shots=1).result()
            memory += result.get_counts()

        return memory
    
    @staticmethod
    def compare_orbits(orbit1: list, orbit2: list, length: int, mode=0) -> int:

        """
            This function is used to compare two orbits, using the 
            Hamming distance metric.

            The idea behind that is to find two states with min or max
            Hamming distance between them.

            This function needed to realize symmetry-invariant version
            of Hamming distance. As orbit consist of every symmetry 
            transformation of representative state, comparing two states
            from different orbits allows one to find symmetry-invariant
            Hamming distance.

            It may be useful changing comparing condition - for example
            to find not min but max distance.

            States are strings.
        """

        result_distance = length

        for s1 in orbit1:
            for s2 in orbit2:
                
                result_distance = min(result_distance, Mutual.hamming_dist(s1, s2, length))

        return result_distance

f"""
    Chain() class contains methods, applicable for magnet system's with
    geometry of a chain. Usually, in this systems two type of symmetries
    are used - translation symmetry and parity symmetry.
"""

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
        unsymm_vec = np.zeros(2 ** num_of_spins)

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

    """
        Translation() and parity() methods are used to obtain result of
        action of symmetry group's element on an input state.
    """

    @staticmethod
    def translation(state: str, num_of_spins: int) -> list:
        result = [state]

        for t in range(1, num_of_spins):
            rolled_state = state[t:] + state[:t]
            result.append(rolled_state)

        return result

    @staticmethod
    def parity(state) -> list:
        return [state, state[::-1]]

    @staticmethod
    def orbit_on_fly(state, num_of_spins, translation=True, parity=True) -> list:

        """
            orbit_on_fly() implements building orbits on fly. In contrast to prepare_orbits(),
            declared further, this method does NOT iterate through all possible system's states, 
            which makes him applicable to big systems.

            TODO optimize parity, try to avoid rolling state second time
        """

        checked = set()
        checked.add(state)

        if translation:
            tmp_list = []
            for checked_state in checked:
                tmp_list += Chain.translation(checked_state, num_of_spins)

            for elem in tmp_list:
                checked.add(elem)

        if parity:
            tmp_list = []
            for checked_state in checked:
                tmp_list += Chain.parity(checked_state)

            for elem in tmp_list:
                checked.add(elem)

        return list(checked)

    @staticmethod
    def prepare_orbits(num_of_spins, states, translation=True, parity=False) -> dict:

        """
            This function is used to build set of states a.k.a orbits, using translation
            and parity symmetry groups for systems with geometry of chain.

            Full symmetry group is an outer product of translation and parity symmetry groups.

            TODO unoptimized, rework later
        """

        num_of_states = 2 ** num_of_spins

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
    def bare_to_binary(bare_result_filename, binary_result_filename, num_of_spins):

        """
            This function converst bare samples, obtained via bare_sample(),
            into binary samples, building orbits on fly and making random.choice()
        """

        visited_orbits = {}

        bare_file = open(bare_result_filename, mode="r")
        bare_samples = bare_file.read().split(';')
        bare_samples.pop(-1)
        bare_file.close()

        bare_samples = list(map(int, bare_samples))

        binary_file = open(binary_result_filename, mode="w")
        
        # TODO убрать как сделаю Изинг !!!!!!!!!!!!!
        checked = set()
        bitstrings_file = open(binary_result_filename[:-4]+'+bst.dat', mode="w")

        for sample in bare_samples:
            if sample not in visited_orbits:
                state = bin(sample)[2:].zfill(num_of_spins)
                visited_orbits[sample] = Chain.orbit_on_fly(state, num_of_spins, translation=True, parity=True)

            choice = random.choice(visited_orbits[sample])
            
            # TODO тоже убирается!!!!
            if choice not in checked:
                print(choice, file=bitstrings_file, end='\n')
                checked.add(choice)

            binary_file.write(choice)
        binary_file.close()

f"""
    Square() class contains methods, applicable for magnet system's with
    geometry of a square. Symmetry methods declared further.
"""

class Square(object):

    @staticmethod
    def translation_x(state: str, square_site: int) -> list:

        # translates square column along x axis

        result = []
        state_length = len(state)
        rolled_state = ""

        for t in range(0, square_site):
            rolled_state = ""
            for i in range(0, state_length, square_site):
                cut_state = state[i:i+square_site]
                rolled_state += cut_state[t:] + cut_state[:t]
            result.append(rolled_state)

        return result

    @staticmethod
    def translation_y(state: str, square_site: int) -> list:

        # translates square row along y-axis

        result = []

        for t in range(0, square_site):
            translation = t * square_site
            rolled_state = state[translation:] + state[:translation]
            result.append(rolled_state)

        return result

    @staticmethod
    def reflection_x(state: str, square_site: int) -> list:

        # reflects square along x axis

        result = [state]
        state_length = len(state)
        reflected_state = ""

        for i in range(0, state_length, square_site):
            reflected_state += state[i:i + square_site][::-1]

        result.append(reflected_state)

        return result

    @staticmethod
    def reflection_y(state: str, square_site: int) -> list:

        # reflects square along y axis

        result = [state]
        state_length = len(state)
        rev_state = state[::-1]
        reflected_state = ""

        for i in range(0, state_length, square_site):
            reflected_state += rev_state[i:i + square_site][::-1]

        result.append(reflected_state)

        return result

    @staticmethod
    def rotation(state: str, square_site: int) -> list:

        # rotates square counterclockwise

        result = [state]
        rotated_state = ""

        for r in range(0, 3):
            tmp_state = result[-1]
            for j in range(1, square_site + 1):
                for i in range(1, square_site + 1):
                    rotated_state += tmp_state[square_site * i - j]

            result.append(rotated_state)
            rotated_state = ""

        return result

    @staticmethod
    def orbit_on_fly(state, square_site, translation_x=True, translation_y=True,
                     reflection_x=True, reflection_y=True, rotation=True) -> list:

        """
            orbit_on_fly() function builds state orbit, using elements of square symmetry group
            and translation symmetry group

            TODO: consider returning binary instead of strings in symmetry operations
        """

        checked = set()
        checked.add(state)

        if translation_x:
            tmp_list = []
            for checked_state in checked:
                tmp_list += Square.translation_x(checked_state, square_site)

            for elem in tmp_list:
                checked.add(elem)

        if translation_y:
            tmp_list = []
            for checked_state in checked:
                tmp_list += Square.translation_y(checked_state, square_site)

            for elem in tmp_list:
                checked.add(elem)

        if reflection_x:
            tmp_list = []
            for checked_state in checked:
                tmp_list += Square.reflection_x(checked_state, square_site)

            for elem in tmp_list:
                checked.add(elem)

        if reflection_y:
            tmp_list = []
            for checked_state in checked:
                tmp_list += Square.reflection_y(checked_state, square_site)

            for elem in tmp_list:
                checked.add(elem)

        if rotation:
            tmp_list = []
            for checked_state in checked:
                tmp_list += Square.rotation(checked_state, square_site)

            for elem in tmp_list:
                checked.add(elem)

        return list(checked)

    @staticmethod
    def bare_to_binary(bare_result_filename, binary_result_filename, num_of_spins):

        """
            The same thing as Chain.bare_to_binary()
        """

        checked = set()
        visited_orbits = {}
        square_site = int(np.sqrt(num_of_spins))

        bare_file = open(bare_result_filename, mode="r")
        bare_samples = bare_file.read().split(';')
        bare_samples.pop(-1)
        bare_file.close()

        # я поменял код. семплится не в строку, а построчною так удобнее
        # ещё отделяются уникальные. так удобнее. если что поменять
        bare_samples = list(map(int, bare_samples))

        # binary_file = open(binary_result_filename, mode="w")
        bitstring_file = open(binary_result_filename, mode="w")
        for sample in bare_samples:
            if sample not in visited_orbits:
                state = bin(sample)[2:].zfill(num_of_spins)
                visited_orbits[sample] = Square.orbit_on_fly(state, square_site)

            choice = random.choice(visited_orbits[sample])
        
            # string cutting %%%

            # new_choice = Mutual.sample_cut(choice, num_of_spins)
            # new_choice = choice[4:]

            # string cutting %%%

            # binary_file.write(choice)
            if choice not in checked:
                print(choice, file=bitstring_file, end='\n')
                checked.add(choice)
                
        bitstring_file.close()

        # print(visited_orbits)

    @staticmethod
    def unsymmetrizer(h5_filename: str, square_site: int) -> np.ndarray:

        """
            unsymmetrizer() function converts eigenvector built in symmetrized basis
            via projectors to and eigenvector in default stationary states basis.

            We set permutation sector equal to zero, so all symmetry characters are 1.

            TODO: F*** BINARIES GO INTEGER!!!
        """

        num_of_spins = square_site**2
        unsymm_vec = np.zeros(2**num_of_spins, dtype='complex')

        # import exact-diagonalization data from h5 file
        data_file = h5py.File(h5_filename, 'r')
        ed_vector = list(data_file['/hamiltonian/eigenvectors'])[0]
        representatives = list(data_file['/basis/representatives'])

        num_of_reps = len(representatives)

        for i in range(0, num_of_reps):

            b_state = bin(representatives[i])[2:].zfill(num_of_spins)
            orbit = Square.orbit_on_fly(b_state, square_site)

            value = ed_vector[i] / np.sqrt(len(orbit))

            for state in orbit:
                unsymm_vec[int(state, 2)] = value

        return unsymm_vec
