import numpy as np
import os

class System:

    def __init__(self, parameter_storage):

        self.parameter_storage      = parameter_storage
        self.acceptor_number        = parameter_storage.parameters["acceptorNumber"]
        self.electrode_number       = parameter_storage.parameters["hoppingSiteNumber"] - parameter_storage.parameters["acceptorNumber"]
        self.hopping_site_number    = parameter_storage.parameters["hoppingSiteNumber"]
        self.loc_len_a              = parameter_storage.parameters["a"]
        self.storing_mode           = parameter_storage.parameters["storingMode"]

        # Arrays
        self.acceptor_positions_x           = np.zeros(self.acceptor_number)
        self.acceptor_positions_y           = np.zeros(self.acceptor_number)
        self.electrode_positions_x          = np.zeros(self.electrode_number)
        self.electrode_positions_y          = np.zeros(self.electrode_number)
        self.occupation                     = [False] * self.acceptor_number
        self.random_energies                = np.zeros(self.acceptor_number)
        self.interaction_partners           = [[] for _ in range(self.acceptor_number)]
        self.hopping_partners_acceptors     = [[] for _ in range(self.hopping_site_number)]
        self.hopping_partners_electrodes    = [[] for _ in range(self.hopping_site_number)]

        # Last swapped
        self.last_swapped1 = 0
        self.last_swapped2 = 0

        # Rates
        self.rates_sum                  = 0
        self.constant_rates_sum_part    = 0

        # Additional data
        self.additional_datafile = None

        # Initialization
        self.initialize_matrices()
        self.get_ready_for_run()
    
    def initialize_matrices(self):

        # Initialize matrices and arrays
        self.pair_energies      = np.zeros((self.acceptor_number, self.acceptor_number))
        self.distances          = np.zeros((self.hopping_site_number, self.hopping_site_number))
        self.delta_energies     = np.zeros((self.hopping_site_number, self.hopping_site_number))
        self.rates              = np.zeros((self.hopping_site_number, self.hopping_site_number))
        self.base_rates         = np.zeros((self.hopping_site_number, self.hopping_site_number))
        self.energies           = np.zeros(self.hopping_site_number)
        self.current_counter    = np.zeros(self.hopping_site_number, dtype=int)

    def create_random_new_device(self):

        for i in range(self.electrode_number):
            self.electrode_positions_x[i] = self.parameter_storage.parameters["radius"] * np.cos(np.radians(self.parameter_storage.electrodes[i].pos))
            self.electrode_positions_y[i] = self.parameter_storage.parameters["radius"] * np.sin(np.radians(self.parameter_storage.electrodes[i].pos))

        iteration = 0
        for i in range(self.acceptor_number):
            iteration = 0
            while iteration <= 10000:
                iteration += 1
                if iteration > 10000:
                    raise ValueError("Could not find device, reduce minDist or acceptorNumber")
                acceptor_pos_x = np.random.uniform(-self.parameter_storage.parameters["radius"], self.parameter_storage.parameters["radius"])
                acceptor_pos_y = np.random.uniform(-self.parameter_storage.parameters["radius"], self.parameter_storage.parameters["radius"])
                if acceptor_pos_x ** 2 + acceptor_pos_y ** 2 > self.parameter_storage.parameters["radius"] ** 2:
                    continue
                if np.min(np.sqrt((acceptor_pos_x - self.acceptor_positions_x[:i]) ** 2 + (acceptor_pos_y - self.acceptor_positions_y[:i]) ** 2)) < self.parameter_storage.parameters["minDist"]:
                    continue
                if np.min(np.sqrt((acceptor_pos_x - self.electrode_positions_x) ** 2 + (acceptor_pos_y - self.electrode_positions_y) ** 2)) < self.parameter_storage.parameters["minDist"]:
                    continue
                self.acceptor_positions_x[i] = acceptor_pos_x
                self.acceptor_positions_y[i] = acceptor_pos_y
                break

        for i in range(self.parameter_storage.parameters["donorNumber"]):
            while True:
                donor_pos_x = np.random.uniform(-self.parameter_storage.parameters["radius"], self.parameter_storage.parameters["radius"])
                donor_pos_y = np.random.uniform(-self.parameter_storage.parameters["radius"], self.parameter_storage.parameters["radius"])
                if donor_pos_x ** 2 + donor_pos_y ** 2 <= self.parameter_storage.parameters["radius"] ** 2:
                    break
            self.donor_positions_x[i] = donor_pos_x
            self.donor_positions_y[i] = donor_pos_y

        std_dev = self.parameter_storage.parameters["randomEnergyStdDev"]
        random_energy_enabled = std_dev != 0.0
        random_energies = np.random.normal(0, std_dev, self.acceptor_number) if random_energy_enabled else np.zeros(self.acceptor_number)
        self.random_energies = random_energies

        e = self.parameter_storage.parameters["e"]
        kT = self.parameter_storage.parameters["kT"]
        device_file_name = os.path.join(self.parameter_storage.working_directory, "device.txt")
        with open(device_file_name, "w") as device_file:
            device_file.write(self.parameter_storage.geometry + "\n")
            device_file.write("acceptors: posX, posY, randomEnergy\n")
            for i in range(self.acceptor_number):
                device_file.write(f"{self.acceptor_positions_x[i] * self.parameter_storage.parameters['R']} "
                                f"{self.acceptor_positions_y[i] * self.parameter_storage.parameters['R']} "
                                f"{random_energies[i] / e * kT}\n")
            device_file.write("\n")
            device_file.write("donors: posX, posY\n")
            for i in range(self.parameter_storage.parameters["donorNumber"]):
                device_file.write(f"{self.donor_positions_x[i] * self.parameter_storage.parameters['R']} "
                                f"{self.donor_positions_y[i] * self.parameter_storage.parameters['R']}\n")

    def get_ready_for_run(self):

        # Set electrodes
        for i in range(self.electrode_number):
            self.electrode_positions_x[i] = self.parameter_storage.parameters["radius"] * math.cos(math.radians(self.parameter_storage.electrodes[i].pos))
            self.electrode_positions_y[i] = self.parameter_storage.parameters["radius"] * math.sin(math.radians(self.parameter_storage.electrodes[i].pos))

        # Init laplace solver
        circle = FiniteElementeCircle(
            self.parameter_storage.parameters["radius"],
            self.parameter_storage.parameters["finiteElementsResolution"]
        )
        for i in range(self.electrode_number):
            circle.setElectrode(
                0,
                self.parameter_storage.electrodes[i].pos / 360 * 2 * math.pi - 0.5 * self.parameter_storage.parameters["electrodeWidth"] / self.parameter_storage.parameters["radius"],
                self.parameter_storage.electrodes[i].pos / 360 * 2 * math.pi + 0.5 * self.parameter_storage.parameters["electrodeWidth"] / self.parameter_storage.parameters["radius"]
            )
        self.finEle = circle
        self.finEle.initRun(True)

        # Calc distances and pair energies
        # acc<->acc
        min_dist = 0.0
        for i in range(self.acceptor_number):
            for j in range(i):
                dist = math.sqrt((self.acceptor_positions_x[i] - self.acceptor_positions_x[j])**2 + (self.acceptor_positions_y[i] - self.acceptor_positions_y[j])**2)
                self.distances[i * self.hopping_site_number + j] = dist
                self.distances[j * self.hopping_site_number + i] = dist
                self.pair_energies[i * self.acceptor_number + j] = -self.parameter_storage.parameters["I0"] / dist
                self.pair_energies[j * self.acceptor_number + i] = -self.parameter_storage.parameters["I0"] / dist
        for i in range(self.acceptor_number):
            self.pair_energies[i * self.acceptor_number + i] = 0
            self.distances[i * self.hopping_site_number + i] = 0
        # acc->el
        for i in range(self.acceptor_number):
            for j in range(self.electrode_number):
                dist = math.sqrt((self.acceptor_positions_x[i] - self.electrode_positions_x[j])**2 + (self.acceptor_positions_y[i] - self.electrode_positions_y[j])**2)
                self.distances[i * self.hopping_site_number + (j + self.acceptor_number)] = dist
        # el->acc
        for i in range(self.electrode_number):
            for j in range(self.acceptor_number):
                dist = math.sqrt((self.electrode_positions_x[i] - self.acceptor_positions_x[j])**2 + (self.electrode_positions_y[i] - self.acceptor_positions_y[j])**2)
                self.distances[(i + self.acceptor_number) * self.hopping_site_number + j] = dist
        # el<->el
        for i in range(self.electrode_number):
            for j in range(i):
                dist = math.sqrt((self.electrode_positions_x[i] - self.electrode_positions_x[j])**2 + (self.electrode_positions_y[i] - self.electrode_positions_y[j])**2)
                self.distances[(i + self.acceptor_number) * self.hopping_site_number + (j + self.acceptor_number)] = dist
                self.distances[(j + self.acceptor_number) * self.hopping_site_number + (i + self.acceptor_number)] = dist
            self.distances[(i + self.acceptor_number) * (self.hopping_site_number + 1)] = 0

        # Set hopping partners and calculate base rates
        low_dist_blocked = 0
        for i in range(self.hopping_site_number):
            self.hopping_partners_acceptors.append([])
            self.hopping_partners_electrodes.append([])
            if (i - self.acceptor_number) not in self.parameter_storage.isolated_electrodes:
                for j in range(self.acceptor_number):
                    if i != j and self.distances[i * self.hopping_site_number + j] < self.parameter_storage.parameters["maxHoppingDist"] and self.distances[i * self.hopping_site_number + j] > self.parameter_storage.parameters["minHoppingDist"]:
                        self.hopping_partners_acceptors[i].append(j)
                        self.base_rates[i * self.hopping_site_number + j] = medium_fast_exp(-2 * self.distances[i * self.hopping_site_number + j] / self.loc_len_a)
                    elif i != j and self.distances[i * self.hopping_site_number + j] < self.parameter_storage.parameters["minHoppingDist"]:
                        low_dist_blocked += 1
                for j in range(self.acceptor_number, self.hopping_site_number):
                    if (j - self.acceptor_number) not in self.parameter_storage.isolated_electrodes:
                        if i != j and self.distances[i * self.hopping_site_number + j] < self.parameter_storage.parameters["maxHoppingDist"] and self.distances[i * self.hopping_site_number + j] > self.parameter_storage.parameters["minHoppingDist"]:
                            self.hopping_partners_electrodes[i].append(j)
                            self.base_rates[i * self.hopping_site_number + j] = medium_fast_exp(-2 * self.distances[i * self.hopping_site_number + j] / self.loc_len_a)
                        elif i != j and self.distances[i * self.hopping_site_number + j] < self.parameter_storage.parameters["minHoppingDist"]:
                            low_dist_blocked += 1

        if low_dist_blocked > 0:
            print("Hopping connections blocked due to small distance:", low_dist_blocked // 2)

        # Set interaction partners
        self.interaction_partners = [[] for _ in range(self.acceptor_number)]
        for i in range(self.acceptor_number):
            for j in range(self.acceptor_number):
                if i != j and self.distances[i * self.hopping_site_number + j] < self.parameter_storage.parameters["maxInteractionDist"]:
                    self.interaction_partners[i].append(j)

        # Set start occupation of acceptors
        occupation_buffer = [False] * self.acceptor_number
        indices_unoccupied = list(range(self.acceptor_number))
        for _ in range(self.acceptor_number - self.parameter_storage.parameters["donorNumber"]):
            index = random.randint(0, len(indices_unoccupied) - 1)
            occupation_buffer[indices_unoccupied[index]] = True
            indices_unoccupied.pop(index)
        self.set_occupation(occupation_buffer)

        # Set electrode energy
        for i in range(self.electrode_number):
            self.energies[i + self.acceptor_number] = 0

        self.ready_for_run = True

    def reset(self):

        self.time = 0
        self.currentCounter = [0] * self.hoppingSiteNumber

    def run(self, steps):
        for i in range(steps):
            self.updateRates()
            self.findSwap()
            self.updateAfterSwap()
            self.increaseTime()

    def updateRates(self):

        self.ratesSum = self.constantRatesSumPart

        # acc acc hopp
        for i in range(self.acceptorNumber):
            if self.occupation[i]:
                for j in self.hoppingPartnersAcceptors[i]:
                    if not self.occupation[j]:
                        delta_energy = self.energies[j] - self.energies[i] + self.pairEnergies[i * self.acceptorNumber + j]
                        if delta_energy <= 0:
                            self.rates[i * self.hoppingSiteNumber + j] = self.baseRates[i * self.hoppingSiteNumber + j]
                        else:
                            self.rates[i * self.hoppingSiteNumber + j] = self.baseRates[i * self.hoppingSiteNumber + j] * enhance.mediumFastExp(-delta_energy)
                        self.ratesSum += self.rates[i * self.hoppingSiteNumber + j]

        # el-acc hopp
        for i in range(self.acceptorNumber, self.hoppingSiteNumber):
            for j in self.hoppingPartnersAcceptors[i]:
                if not self.occupation[j]:
                    delta_energy = self.energies[j] - self.energies[i]
                    if delta_energy <= 0:
                        self.rates[i * self.hoppingSiteNumber + j] = self.baseRates[i * self.hoppingSiteNumber + j]
                    else:
                        self.rates[i * self.hoppingSiteNumber + j] = self.baseRates[i * self.hoppingSiteNumber + j] * enhance.mediumFastExp(-delta_energy)
                    self.ratesSum += self.rates[i * self.hoppingSiteNumber + j]

        # acc-el hopp
        for i in range(self.acceptorNumber):
            if self.occupation[i]:
                for j in self.hoppingPartnersElectrodes[i]:
                    delta_energy = self.energies[j] - self.energies[i]
                    if delta_energy <= 0:
                        self.rates[i * self.hoppingSiteNumber + j] = self.baseRates[i * self.hoppingSiteNumber + j]
                    else:
                        self.rates[i * self.hoppingSiteNumber + j] = self.baseRates[i * self.hoppingSiteNumber + j] * enhance.mediumFastExp(-delta_energy)
                    self.ratesSum += self.rates[i * self.hoppingSiteNumber + j]

    def findSwap(self):

        # noSwapFound:
        rndNumber = enhance.random_double(0, self.ratesSum)
        partRatesSum = 0

        # from acc ...
        for i in range(self.acceptorNumber):
            if self.occupation[i]:
                # .. to acc
                for j in self.hoppingPartnersAcceptors[i]:
                    if not self.occupation[j]:
                        partRatesSum += self.rates[i * self.hoppingSiteNumber + j]
                        if partRatesSum > rndNumber:
                            self.lastSwapped1 = i
                            self.lastSwapped2 = j
                            return
                # .. to electrode
                for j in self.hoppingPartnersElectrodes[i]:
                    partRatesSum += self.rates[i * self.hoppingSiteNumber + j]
                    if partRatesSum > rndNumber:
                        self.lastSwapped1 = i
                        self.lastSwapped2 = j
                        return

        # from electrode ...
        for i in range(self.acceptorNumber, self.hoppingSiteNumber):
            # .. to acc
            for j in self.hoppingPartnersAcceptors[i]:
                if not self.occupation[j]:
                    partRatesSum += self.rates[i * self.hoppingSiteNumber + j]
                    if partRatesSum > rndNumber:
                        self.lastSwapped1 = i
                        self.lastSwapped2 = j
                        return
            # .. to electrode
            for j in self.hoppingPartnersElectrodes[i]:
                partRatesSum += self.rates[i * self.hoppingSiteNumber + j]
                if partRatesSum > rndNumber:
                    self.lastSwapped1 = i
                    self.lastSwapped2 = j
                    return

        raise ValueError("Internal error! No swap found.")

    def updateAfterSwap(self):

        self.currentCounter[self.lastSwapped1] -= 1
        self.currentCounter[self.lastSwapped2] += 1

        if self.lastSwapped1 < self.acceptorNumber:  # last swapped1 = acceptor, else electrode
            self.occupation[self.lastSwapped1] = False
            # update energy
            for j in self.interactionPartners[self.lastSwapped1]:
                self.energies[j] += self.pairEnergies[self.lastSwapped1 * self.acceptorNumber + j]

        if self.lastSwapped2 < self.acceptorNumber:  # last swapped2 = acceptor
            self.occupation[self.lastSwapped2] = True
            for j in self.interactionPartners[self.lastSwapped2]:
                self.energies[j] -= self.pairEnergies[self.lastSwapped2 * self.acceptorNumber + j]

    def updatePotential(self, voltages):
        # Reset potential
        self.resetPotential()

        # Recalculate potential
        for i in range(self.electrodeNumber):
            self.finEle.updateElectrodeVoltage(i, voltages[i])

        self.finEle.run()

        # Set new potential
        self.setNewPotential()

        if self.parameterStorage.verbose:
            self.parameterStorage.parameters["additionalFileNumber"] += 1
            additional_file = DataFile(
                self.parameterStorage.workingDirecotry + "additionalData" +
                str(round(self.parameterStorage.parameters["additionalFileNumber"])) +
                ".hdf5", True)
            additional_file.createDataset("time", [1])
            additional_file.createDataset("lastSwapp", [2])
            additional_file.createDataset("occupation", [self.acceptorNumber])
            additional_file.createDataset("energies", [self.acceptorNumber])

    def set_occupation(self, new_occupation):

        # Set occupation
        self.occupation = new_occupation

        # Set acceptor energies
        for i in range(self.acceptor_number):
            energy = self.finEle.getPotential(self.acceptor_positions_x[i], self.acceptor_positions_y[i]) * self.parameter_storage.parameters["e"] / self.parameter_storage.parameters["kT"]
            for j in range(self.parameter_storage.parameters["donorNumber"]):
                energy += self.parameter_storage.parameters["I0"] * 1 / math.sqrt((self.acceptor_positions_x[i] - self.donor_positions_x[j])**2 + (self.acceptor_positions_y[i] - self.donor_positions_y[j])**2)
            self.energies[i] = energy

        for i in range(self.acceptor_number):
            self.energies[i] += self.random_energies[i]

        # Set Coulomb part (with start occupation)
        for i in range(self.acceptor_number):
            if not self.occupation[i]:
                for j in self.interaction_partners[i]:
                    self.energies[j] += self.pair_energies[i * self.acceptor_number + j]

    def increaseTime(self):
        self.time += np.log(random.uniform(0, 1)) / (-1 * self.ratesSum)  # Avoid zero

    def resetPotential(self):
        # Reset old potential
        for i in range(self.acceptorNumber):
            energy = self.finEle.getPotential(self.acceptorPositionsX[i], self.acceptorPositionsY[i]) * self.parameterStorage.parameters["e"] / self.parameterStorage.parameters["kT"]
            self.energies[i] -= energy

        for i in range(self.electrodeNumber):
            energy = self.finEle.getPotential(self.electrodePositionsX[i], self.electrodePositionsY[i]) * self.parameterStorage.parameters["e"] / self.parameterStorage.parameters["kT"]
            self.energies[i + self.acceptorNumber] -= energy

    def setNewPotential(self):
        # Set new potential
        for i in range(self.acceptorNumber):
            energy = self.finEle.getPotential(self.acceptorPositionsX[i], self.acceptorPositionsY[i]) * self.parameterStorage.parameters["e"] / self.parameterStorage.parameters["kT"]
            self.energies[i] += energy

        for i in range(self.electrodeNumber):
            energy = self.finEle.getPotential(self.electrodePositionsX[i], self.electrodePositionsY[i]) * self.parameterStorage.parameters["e"] / self.parameterStorage.parameters["kT"]
            self.energies[i + self.acceptorNumber] += energy

        # Set deltaEnergies and rates (only constant part = el-el interaction)
        # Init all to zero
        for i in range(self.acceptorNumber, self.hoppingSiteNumber):
            for j in range(self.acceptorNumber, i):
                self.deltaEnergies[i * self.hoppingSiteNumber + j] = 0
                self.deltaEnergies[j * self.hoppingSiteNumber + i] = 0
                self.rates[i * self.hoppingSiteNumber + j] = 0
                self.rates[j * self.hoppingSiteNumber + i] = 0

        self.constantRatesSumPart = 0
        for i in range(self.acceptorNumber, self.hoppingSiteNumber):
            for j in self.hoppingPartnersElectrodes[i]:
                self.deltaEnergies[i * self.hoppingSiteNumber + j] = self.energies[j] - self.energies[i]

                if self.deltaEnergies[i * self.hoppingSiteNumber + j] < 0:
                    self.rates[i * self.hoppingSiteNumber + j] = self.baseRates[i * self.hoppingSiteNumber + j]
                elif self.deltaEnergies[i * self.hoppingSiteNumber + j] > 0:
                    self.rates[i * self.hoppingSiteNumber + j] = self.baseRates[i * self.hoppingSiteNumber + j] * enhance.mediumFastExp(-self.deltaEnergies[i * self.hoppingSiteNumber + j])
                else:
                    self.rates[i * self.hoppingSiteNumber + j] = self.baseRates[i * self.hoppingSiteNumber + j]

                self.constantRatesSumPart += self.rates[i * self.hoppingSiteNumber + j]

