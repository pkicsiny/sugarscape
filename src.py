import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shutil


def getmap(i):
    """
    Load map from txt file.
    :param i: int, ID of map
    :return map_: numpy array
    :return mapDim: int, dimension of map
    """
    
    map_ = np.loadtxt('maps/map' + str(i) + '.txt')
    mapDim = len(map_)
    return map_, mapDim


def checkInputs(input_dict):
    """
    Check correctness of simulation inputs.
    :param input_dict: dictionary:
        mapDim: int, dimension of map
        nAgents: int, number of agents
        maxAge: int, age of agents
        simSteps: int, number of simulation steps
        scoutRange: int, vision distance of Agentss in cells
        consumptionRate: int, rate of consuming coins
        initWealth: int, initial wealth
        reproductionCost: int, cost of reproduction
        plotFreq: int, rate of plotting wealth distribution
    :return: 1 if all input checks pass, assertion error otherwise
    """
    
    input_to_text = {"nAgents": "Number of agents",
                     "maxAge": "Maximum age of agents",
                     "scoutRange": "Vision distance",
                     "consumptionRate": "Consumption rate",
                     "refillRate": "Refill rate",
                     "initWealth": "Initial wealth",
                     "simSteps": "Number of steps",
                     "reproductionCost": "Reproduction cost"}
    
    # upper limits on inputs
    nAgents_lim = int(input_dict["mapDim"])**2
    maxAge_lim = 10000
    scoutRange_lim = int(input_dict["mapDim"])
    consumptionRate_lim = 10
    refillRate_lim = 100
    initWealth_lim = 100
    simSteps_lim = 10000
    reproductionCost_lim = 100
    
    # checks
    for (key, value) in input_dict.items():
        
        try:
            input_dict[key] = int(value)
        except:
            return '{}\nmust be an integer!'.format(input_to_text[key])
            
        if int(value) < 0:
            return '{}\nmust be positive!'.format(input_to_text[key])
    
    if not (input_dict["nAgents"] > 0 and input_dict["nAgents"] < nAgents_lim):
         return 'Number of agents must be nonzero\nand smaller than {}!'.format(nAgents_lim)
    if not input_dict["maxAge"] < maxAge_lim:
         return 'Maximum age must be \nsmaller than {}!'.format(maxAge_lim)
    if not (input_dict["scoutRange"] > 0 and input_dict["scoutRange"] < scoutRange_lim):
         return 'Vision distance must be nonzero\nand smaller than {}!'.format(scoutRange_lim)
    if not input_dict["consumptionRate"] <= consumptionRate_lim:
         return 'Consumption rate must be\nmaximum {}!'.format(consumptionRate_lim)
    if not (input_dict["refillRate"] > 0 and input_dict["refillRate"] <= refillRate_lim):
         return 'Refill rate must be nonzero\nand maximum {}!'.format(refillRate_lim)
    if not input_dict["initWealth"] <= initWealth_lim:
         return 'Initial wealth must be\nmaximum {}!'.format(initWealth_lim)
    if not (input_dict["simSteps"] > 0 and input_dict["simSteps"] <= simSteps_lim):
         return 'Number of steps must be nonzero\nand maximum {}!'.format(simSteps_lim)
    if not input_dict["reproductionCost"] <= reproductionCost_lim:
         return 'Reproduction cost must be\nmaximum {}!'.format(reproductionCost_lim)
    return 0


def getCoordinates(input_dict):
    """
    Generates >>nAgents<< random complex numbers that are the agents' starting corrdinates on the map.
    :return agent_coords: numpy array of compplex numbers
    """
    
    agent_coords = []
    while len(agent_coords) != input_dict["nAgents"]:
        
        # generate a new coordinate
        agent_coords.append(np.random.randint(0, input_dict["mapDim"]) + np.random.randint(0, input_dict["mapDim"]) * 1j)
        
        # regenerate while coord is not unique
        while len(agent_coords) != len(set(agent_coords)):
            agent_coords[-1] = np.random.randint(0, input_dict["mapDim"]) + np.random.randint(0, input_dict["mapDim"]) * 1j
    return np.array(agent_coords).reshape((1, input_dict["nAgents"]))


# agentek lekerulnek a mapra es felveszik az ottani coinokat, -1 lesz a helyen
def spawnAgents(agent_coords, map_, input_dict, gender_vec):
    """
    Spawn agents onto the map and take coins that are at the spawn spot.
    :param agent_coords: numpy array of shape (1 x #agents) containing complex numbers representing agent coordinates
    :return agent_spawn_gains: numpy array of shape (1 x #agents) of initial wealth of spawned agents
    :param gender_vec: numpy array of ints if size (1 x nAgents) with agent gender
    (1: male, 0: female, -1: neutral)
    :return map_: updated map
    """
    
    # maximum number of coins on a cell (global max)
    maxCoin = int(np.amax(map_))
    
    # array for initial coin gains
    agent_spawn_gains = np.zeros(agent_coords.shape[1], dtype=int).reshape((1, input_dict["nAgents"]))

    # loop over agent spawn coordinates
    for i in range(agent_coords.shape[1]):
    
            # gain coins on spawn cell
            agent_spawn_gains[0, i] += map_[int(agent_coords[0, i].real), int(agent_coords[0, i].imag)]
                
            # set spawn spot value to -3 (agent is on that cell)
            # set cell value according to agent gender
            # no gender
            if gender_vec[0] == -1:
                map_[int(agent_coords[0, i].real), int(agent_coords[0, i].imag)] = -3
            else:

                # male
                if gender_vec[i]:
                    map_[int(agent_coords[0, i].real), int(agent_coords[0, i].imag)] = -31
                       
                # female
                else:
                    map_[int(agent_coords[0, i].real), int(agent_coords[0, i].imag)] = -30

    return agent_spawn_gains, map_


def lookAround(agent_coord, map_, scoutRange):
    """
    :param agent_coord: complex number, coordinate of a single agent
    :param map_: map
    :param scoutRange: int, vision distance
    :return scout_dict: dict containing possible next steps for an agent
    """
    
    scout_dict = {"coord": np.zeros((scoutRange, 4), dtype=complex),
                  "val": np.zeros((scoutRange, 4), dtype=int)}
    
    # scoutRange x 4 (look in 4 directions)
    scout_dict["val"] = np.zeros((scoutRange, 4), dtype=int)  # ezert 0nak latja a -0.5oket mert nem float, ezert lehet csikicsukizni
    scout_dict["coord"] = np.zeros((scoutRange, 4), dtype=complex)
    
    # look around from close to far until >>scoutRange<<
    for dist in range(1, scoutRange + 1):
        
        # look north
        if agent_coord.real - dist >= 0:
            scout_dict["val"][dist - 1][0] += map_[int(agent_coord.real - dist), int(agent_coord.imag)]
            scout_dict["coord"][dist - 1][0] += agent_coord - dist
        else: # outside map boundary
            scout_dict["val"][dist - 1][0] += -2
            scout_dict["coord"][dist - 1][0] += -2
        
        # look east
        if agent_coord.imag + dist< map_.shape[1]:
            scout_dict["val"][dist - 1][1] += map_[int(agent_coord.real), int(agent_coord.imag + dist)]
            scout_dict["coord"][dist - 1][1] += agent_coord + dist * 1j
        else: # outside map boundary
            scout_dict["val"][dist - 1][1] += -2 
            scout_dict["coord"][dist - 1][1] += -2
        
        # look south
        if agent_coord.real + dist < map_.shape[0]:
            scout_dict["val"][dist - 1][2] += map_[int(agent_coord.real + dist), int(agent_coord.imag)]
            scout_dict["coord"][dist - 1][2] += agent_coord + dist
        else: # outside map boundary
            scout_dict["val"][dist - 1][2] += -2
            scout_dict["coord"][dist - 1][2] += -2

        # look west
        if agent_coord.imag - dist>= 0:
            scout_dict["val"][dist - 1][3] += map_[int(agent_coord.real), int(agent_coord.imag - dist)]
            scout_dict["coord"][dist - 1][3] += agent_coord - dist* 1j
        else: # outside map boundary
            scout_dict["val"][dist - 1][3] += -2
            scout_dict["coord"][dist - 1][3] += -2
            
    return scout_dict


def reproduce(agent_id, agent_coord_vec, agent_wealth_vec,
             gender_vec, attractivity_vec, scout_dict, reproduction_cost,
             fertility_vec, offspring_coords, offspring_wealth, offspring_abortion):
    """
    Agents can reproduce if a male and female are next to each other.
    :param agent_id: int, id of a single agent
    :param agent_coord_vec: numpy array of complex of size (1 x nAgents) with agent coordinates in current step
    :param agent_wealth_vec: numpy array of  ints of size (1 x nAgents) with agent wealth in current step
    :param gender_vec: numpy array of ints if size (1 x nAgents) with agent gender
    (1: male, 0: female, -1: neutral)
    :param attractivity_vec: numpy array of floats of size (1 x nAgents) with agent attractivity
    :param scout_dict: dict containing visible cells by a single agent. Keys: coords, vals. Values: numpy arrays
    :param reproduction_cost: int, a mating couple has to have this many coins together to be able to reproduce
    :param fertility_vec: numpy array of ints if size (1 x nAgents) with agent fertility
    (1: mature, 0: not mature, -1: pregnant)
    :param offspring_coords: list of complex containing offspring spawn coordinates
    :param offspring_wealth: list of int containing offspring initial wealth
    :param offspring_abortion: list of 1 or 0 if offspring has to be aborted
    (happens if mother cannot leave current cell)
    :return: fertility_vec, offspring_coords, offspring_coords, offspring_abortion
    """
    
    # only female agents reproduce
    if not gender_vec[agent_id]:
        
        # collect and rank neighboring male agents
        neighbor_dirs = np.where(scout_dict["val"][0] == -11)
        neighbor_coords = [c for (idx, c) in enumerate(scout_dict["coord"][0]) \
                           if idx in neighbor_dirs[0]]
        neighbor_ids = [idx for (idx, c) in enumerate(agent_coord_vec) \
                        if c in neighbor_coords]
        neighbor_attractivity = [a for (idx, a) in enumerate(attractivity_vec) \
                                 if idx in neighbor_ids]
        neighbor_ids_ranked = [neighbor_ids[n] for n in np.argsort(neighbor_attractivity)][::-1]

        # try to mate with neighbors
        for n in neighbor_ids_ranked:

            # if neighbor is male
            # if male is attractive
            # if female is attractive
            # if combined wealth is enough
            # if both are mature (alive for at least 20 rounds + female has not reproduced yet in current round)
            if gender_vec[n] \
            and np.random.uniform() < attractivity_vec[n] \
            and np.random.uniform() < attractivity_vec[agent_id] \
            and agent_wealth_vec[n] + agent_wealth_vec[agent_id] >= reproduction_cost \
            and fertility_vec[n] == 1 \
            and fertility_vec[agent_id] == 1:
                
                # offspring spawns at mother's coord with a wealth average of the parents
                offspring_coords.append(agent_coord_vec[agent_id])
                offspring_wealth.append(int(np.ceil(
                    .5*(agent_wealth_vec[n] + agent_wealth_vec[agent_id]))))
                
                # default option is abortion
                # (will happen if mother cannot move to new cell e.g. dies or map too busy)
                offspring_abortion.append(1)
                
                # mother is pregnant, exit loop, one agent can reproduce only once per round
                fertility_vec[agent_id] = -1
                break
        
    return fertility_vec, offspring_coords, offspring_coords, offspring_abortion


def makePrefList(scout_dict, consumption_rate, agent_coord):
    """
    Creates preference list of a single agent.
    Preferred step is the closest cell with the most coins in the scouted area.
    :param agent_coord: coordinates of a single agent
    :param scout_dict: dict containing possible next steps for an agent
    :param consumption_rate: int, rate of consuming coins
    :return pref_coords_with_dist: numpy array referring to coordinates in the original map. Order in list is the preference
    for next step of the agent.
    :return sorted_costs: list of the associated costs of next steps
    """
    
    temp_scout_dict = np.zeros_like(scout_dict["val"], dtype=int) 
    temp_scout_dict += scout_dict["val"]
    
    # create empty stack for preferred coordinates and their cost
    pref_list = np.zeros([1, 2], dtype=int)
    pref_list = np.delete(pref_list, 0, axis=0)
    cost_list = []
    
    # sort preferences and disable corresponding cells; go until all cells are disabled
    while np.amax(temp_scout_dict) > -1:
        
        # find coordinates and value of first global maximum
        pref_dist, pref_dir = np.unravel_index(temp_scout_dict.argmax(), temp_scout_dict.shape)
        max_elem = np.amax(temp_scout_dict)
        
        # collect all cells with a value of >>max_elem<<; start from closest neighbors and go until full scout distance
        while pref_dist < len(temp_scout_dict):
            
            # cells in distance >>pref_dist<< in four directions
            row = temp_scout_dict[pref_dist]
            
            # coordinates of maximum value(s) at current distance, shuffled into preferred order
            row_maxes = np.argwhere(row == max_elem)
            np.random.shuffle(row_maxes)
            
            if len(row_maxes) > 0:
            
                # append to preference list and disable already considered cells
                for pref_dir in row_maxes:
                    pref_list = np.append(pref_list, [[pref_dist, int(pref_dir)]], axis=0)
                    
                    # consumption is /distance/round
                    cost_list = np.append(cost_list, (pref_dist + 1) * consumption_rate)
                    temp_scout_dict[pref_dist][int(pref_dir)] = -999
                
            pref_dist += 1
            
    # get ordered list of preferred steps and values, mapped back to original map
    pref_coords = np.array([scout_dict["coord"][pref_list[i, 0], pref_list[i, 1]] for i in range(len(pref_list))], dtype=complex)
    pref_vals = np.array([scout_dict["val"][pref_list[i, 0], pref_list[i, 1]] for i in range(len(pref_list))], dtype=int)
    
    # final gain: gain - cost: coin on next cell - digestion rate
    pref_vals_with_cost = pref_vals - cost_list
        
    # the result arrays
    pref_coords_with_dist = np.zeros(len(pref_list), dtype=complex)
    sorted_costs = np.zeros(len(cost_list), dtype=int)
            
    # reorder preference list according to final gain (distance of next cell included)
    sorted_indices = sorted(range(len(pref_vals_with_cost)), key=lambda i: pref_vals_with_cost[i], reverse=True)
    for i in range(len(pref_coords_with_dist)):
        pref_coords_with_dist[i] += pref_coords[int(sorted_indices[i])]
        sorted_costs[i] += cost_list[int(sorted_indices[i])]
        
    # last ooption is always to stay rest with *consumption_rate* cost
    pref_coords_with_dist = np.append(pref_coords_with_dist, [agent_coord])
    sorted_costs = np.append(sorted_costs, [consumption_rate])
    
    return pref_coords_with_dist, sorted_costs


def moveAgent(map_, pref_coords_with_dist, current_coord, current_wealth, pref_idx, sorted_costs):
    """
    Returns next position and wealth of a single agent.
    :param map_: map
    :param pref_coords_with_dist: numpy array referring to coordinates in the original map. Order in list is the preference
    for next step of the agent.
    :param current_coord: agent's current coordinate
    :param current_wealth: int, current wealth of agent i
    :param pref_idx: int, index in preference list
    :param sorted_costs: list of the associated costs of next steps
    :return new_coord: complex number representing the new coordinate on the map
    :return new_gain: int, number of coins on the new cell
    :return new_wealth: int, new wealth of agent after moving to new cell
    :return is_dead: int, 1 if agent dies, 0 otherwise
    """
    
    is_dead = 0

    # update agent's coordinate on map
    new_coord = pref_coords_with_dist[pref_idx]

    # if agent stays in same place
    if current_coord == new_coord:
    
        # just the consumption rate (current cell value is -1)
        new_gain = -sorted_costs[pref_idx]
    else:
        # update agent's wealth: new gain minus cost of moving to new cell
        new_gain = int(map_[int(pref_coords_with_dist[pref_idx].real)][int(pref_coords_with_dist[pref_idx].imag)]) - sorted_costs[pref_idx]
    
    # new wealth
    new_wealth = current_wealth + new_gain
    
    # if wealth goes below 0, agent dies here
    if new_wealth < 0 and not np.isnan(new_wealth):
        is_dead += 1
        new_wealth = np.nan
    
    return new_coord, new_gain, new_wealth, is_dead


def updateMap(map_, agent_coords, step, map_base, dead_agent_id_list, gender_vec, map_refill, refill_rate=1):
    """
    Updates map. First grow coin then move agents.
    :param map_: state of map after step number >>step<<
    :param agent_coords: numpy array of shape (#steps x #agents)
    :param step: int, current iteration number
    :param map_base: numpy array of starting map
    :param dead_agent_id_list: list of complex numbers, dying coordinates of dead agents
    :param gender_vec: numpy array of ints if size (1 x nAgents) with agent gender
    (1: male, 0: female, -1: neutral)
    :param map_refill: state of refill map after step number >>step<<
    :param refill_rate: int, refill rate of cells. Coins grow by 1 in every *refill_rate*th round after they are freed up.
    :return map_: new map after moving agents
    :return map_refill: new refill map after moving agents
    """
    
    # refill: number of coins on an empty cell grows by 1 in each step, until it reaches its original value
    for row in range(map_.shape[0]):
        for col in range(map_.shape[1]):
        
            # consider cell refill rate i.e. cells grow in every i.th round
            if step > 0 and map_[row, col] < map_base[row, col] and map_[row, col] >= 0 and map_refill[row, col] >=0:
                
                # increment refill cell
                map_refill[row, col] += 1
                
                # if refill cell value matches growth rate, add a coin to map cell
                if map_refill[row, col]%refill_rate == 0:
                    map_[row, col] += 1

    # loop over agents
    for ag in range(agent_coords.shape[1]):
    
        # only living agents
        if ag not in dead_agent_id_list:
            
            # new offspring agents are not spawned at initialization and they have only one valid coordinate: set spawn cell to -3
            if np.isnan(agent_coords[0, ag]) and np.count_nonzero(~np.isnan(agent_coords[:, ag])) == 1:
                
                # set cell value according to agent gender
                # no gender
                if gender_vec[0] == -1:
                    map_[int(agent_coords[step + 1, ag].real), int(agent_coords[step + 1, ag].imag)] = -3
                else:

                    # male
                    if gender_vec[ag]:
                        map_[int(agent_coords[step + 1, ag].real), int(agent_coords[step + 1, ag].imag)] = -31
                       
                    # female
                    else:
                        map_[int(agent_coords[step + 1, ag].real), int(agent_coords[step + 1, ag].imag)] = -30
            else:
                # set cell to 0 that agent has left in current step // start refill counter
                map_[int(agent_coords[step, ag].real), int(agent_coords[step, ag].imag)] = 0
                map_refill[int(agent_coords[step, ag].real), int(agent_coords[step, ag].imag)] = 0
                
                # newly occupied cell is set to -1 
                # set cell value according to agent gender
                # no gender
                if gender_vec[0] == -1:
                    map_[int(agent_coords[step + 1, ag].real), int(agent_coords[step + 1, ag].imag)] = -1
                else:

                    # male
                    if gender_vec[ag]:
                        map_[int(agent_coords[step + 1, ag].real), int(agent_coords[step + 1, ag].imag)] = -11
                       
                    # female
                    else:
                        map_[int(agent_coords[step + 1, ag].real), int(agent_coords[step + 1, ag].imag)] = -10

            # reset refill counter for newly occupied cell
            map_refill[int(agent_coords[step + 1, ag].real), int(agent_coords[step + 1, ag].imag)] = -1
            
        # dead agents
        else:

            # agent arrived to new cell but dies there
            if not np.isnan(agent_coords[step + 1, ag]):
            
                # set cell to 0 that agent has left in current step // start refill counter
                map_[int(agent_coords[step, ag].real), int(agent_coords[step, ag].imag)] = 0
                map_refill[int(agent_coords[step, ag].real), int(agent_coords[step, ag].imag)] = 0
                
                # new cell is set to -2 in this round (occupied by corpse) // reset refill counter
                # set cell value according to agent gender
                # no gender
                if gender_vec[0] == -1:
                    map_[int(agent_coords[step + 1, ag].real), int(agent_coords[step + 1, ag].imag)] = -2
                else:

                    # male
                    if gender_vec[ag]:
                        map_[int(agent_coords[step + 1, ag].real), int(agent_coords[step + 1, ag].imag)] = -21
                       
                    # female
                    else:
                        map_[int(agent_coords[step + 1, ag].real), int(agent_coords[step + 1, ag].imag)] = -20
                map_refill[int(agent_coords[step + 1, ag].real), int(agent_coords[step + 1, ag].imag)] = -1
            
            # agent died in previous step
            if step > 0 and np.isnan(agent_coords[step + 1, ag]) and agent_coords[step, ag] >= 0:
                
                # dispose of dead agent and set cell to 0 // start refill counter
                map_[int(agent_coords[step, ag].real), int(agent_coords[step, ag].imag)] = 0
                map_refill[int(agent_coords[step, ag].real), int(agent_coords[step, ag].imag)] = 0
    return map_, map_refill


def plotWealthDist(wealth, step, plotFreq, save_plots):
    """
    Plots wealth distribution histogram.
    :param agent_wealth: list containing current wealth of each agent after the >>step<<th iteration
    :param step: int, current iteration number
    :param plotFreq: int, rate of plotting wealth distribution
    :param save_plots: bool, whether or not to save plots into /plots folder
    """
    
    # make plot in every >>step<<th iteration
    if step % plotFreq == 0:
        
        # update x axis limits (wealth is maximized in 1000 coins)
        x_lim_max = np.nanmax(wealth) if not all(np.isnan(wealth)) else 100
        xticks = np.linspace(0, 11*np.maximum(np.ceil(.1*x_lim_max), 1), 12)
        
        # create histogram base
        wealth_dist = np.zeros(len(xticks), dtype=int)
        
        # fill up histogram with wealth of living agents
        for d in range(len(wealth)):
            if not np.isnan(wealth[d]):
                for i in range(len(xticks)):
                    if wealth[d] >= xticks[i - 1] and wealth[d] < xticks[i]:
                        wealth_dist[i - 1] += 1
              
        # plot
        color_alive = "b"
        x_bin_width = xticks[1] - xticks[0]
        plt.figure()
        plt.bar(xticks, wealth_dist, width=1 * x_bin_width, color=color_alive, align='edge')
        plt.grid(alpha=.5)
        
        # labels
        if step == 0:
            plt.title('Initial wealth distribution', fontsize=18)
        else:
            plt.title('Wealth distribution after ' + str(step) + ' steps', fontsize=18)
        #label = mpatches.Patch(color=color_alive, label='Living agents')
        #plt.legend(handles=[label])
        plt.xlabel('Wealth', fontsize=14)
        plt.ylabel('Number of living agents', fontsize=14)
        
        # ticks
        plt.xticks(xticks)
        y_lim_max = 1.5*np.amax(wealth_dist)
        yticks = np.linspace(0, 10*np.ceil(.1*y_lim_max), 6) if np.max(wealth_dist) > 1 else [0, 1]
        plt.yticks(yticks)
        plt.ylim(0, y_lim_max)
        
        # save and close
        if save_plots:
            plt.savefig('plots/Wealth_dist_{}.png'.format(step/plotFreq))
        plt.close()


def plotLivingAgentCount(n_living_agents, n_births, n_deaths, step, plotFreq, save_plots):
    """
    Plots the number of living agents agains simulation time.
    :param n_living_agents: list containing current number of living agents after the >>step<<th iteration
    :param n_births: list containing number of agents born in the >>step<<th iteration
    :param n_deaths: list containing number of agents died in the >>step<<th iteration
    :param step: int, current iteration number
    :param plotFreq: int, rate of plotting wealth distribution
    :param save_plots: bool, whether or not to save plots into /plots folder
    """
        
    # make plot in every >>step<<th iteration
    if step % plotFreq == 0:
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))
        
        # top subplot
        ax[0].plot(range(len(n_living_agents)), n_living_agents, c="b")
        #ax[0].scatter(range(len(n_living_agents)), n_living_agents, c="b", marker="o")
        ax[0].grid(alpha=.5)
        ax[0].set_xlabel('Simulation step', fontsize=14)
        ax[0].set_ylabel('Number of living agents', fontsize=14)
        
        # bottom subplot
        ax[1].plot(range(len(n_births)), n_births, c="g", label="birth")
        #ax[1].scatter(range(len(n_births)), n_births, c="g", marker="o", label="birth")
        ax[1].plot(range(len(n_deaths)), n_deaths, c="r", label="death")
        #ax[1].scatter(range(len(n_deaths)), n_deaths, c="r", marker="o", label="death")
        ax[1].grid(alpha=.5)
        ax[1].legend(fontsize=14)
        ax[1].set_xlabel('Simulation step', fontsize=14)
        ax[1].set_ylabel('Number of births and deaths', fontsize=14)
        
        if step == 0:
            ax[0].set_title('Agent demographics after spawn', fontsize=18)
        else:
            ax[0].set_title('Agent demographics after ' + str(step) + ' steps', fontsize=18)
        
        # save and close
        if save_plots:
            plt.savefig('plots/Living_agents_{}.png'.format(step/plotFreq))
        plt.close()      


def generateMap(size, max_coin, map_name, map_type = "random", modes = 1, slope = 2, jag_p = .5):
    """
    :param size: int, map dimension
    :param max_coin: int, highest map cell value
    :param map_name: string, name of the map txt file
    :param map_type: string, specific map types are available:
        "flat": each map cell has *max_coin* value
        "random": random numbers up to *max_coin* for each map cell
        "hills": pyramid-like map landscape. Hills are built up in a bottom-up fashion. 
        Further options for this map type:
    :param modes: int, number of hill-like structures on map
    :param slope: int, slope of hills i.e. no. of neighboring cells with the 
    same value before an increment. E.g. 0,0,1,1,2,2 has a slope of 2.
    :param jag_p: float, between 0 and 1, jaggedness parameter. 0: each hill is
    completely pyramid-like. 1: the hill surface is jagged i.e. the increment is not smooth in
    all directions. In general the probability of not incrementing a cell is .5**jag_p*.
    """
    
    # create maps directory
    if not os.path.isdir("maps"):
        os.makedirs("maps") 
    
    # standard map types
    map_types = ["flat", "random", "hills"]
    
    # check inputs
    if not isinstance(map_name, str):
        return '{}\nmust be a string!'.format(map_name)
    if not (isinstance(map_type, str) and map_type in map_types):
        return '{}\nmust be "flat", "random" or "hills"!'.format(map_type) 
    
    for inp in [size, max_coin, modes, slope]:
        if not isinstance(inp, int):
            return '{}\nmust be an integer!'.format(inp) 
        if int(inp) < 0:
            return '{}\nmust be positive!'.format(inp)
    
    if not (size > 0 and size <= 30):
        return 'Map size must be nonzero\nand at most {}!'.format(30)
    if not (max_coin <= 5):
        return 'Max. number of coins on a\ncell must be at most {}!'.format(5)
    if not (modes > 0 and modes <= 50):
        return 'Number of modes must be nonzero\nand at most {}!'.format(50)
    if not (slope > 0 and slope <= 10):
        return 'Hill slope must be nonzero\nand at most {}!'.format(10)
    
    
    # init map
    base = np.zeros((size, size), dtype=int)
    
    # flat
    if map_type == "flat":
        base += max_coin
    
    # random
    elif map_type == "random":
        base = np.random.randint(max_coin+1, size=(size, size))
    
    # hills
    elif map_type == "hills":
        for m in range(modes):
            
            # select random cell
            x = np.random.randint(size)
            y= np.random.randint(size)
            
            # build hill around selected cell
            neighbor = max_coin*slope
            while neighbor > 0:
                
                # get slice of map
                chunk = base[np.maximum(x-neighbor+1, 0):np.minimum(x+neighbor, size),
                             np.maximum(y-neighbor+1, 0):np.minimum(y+neighbor, size)]
                
                for row in range(np.shape(chunk)[0]):
                    for col in range(np.shape(chunk)[1]):
                    
                        # increment each cell with a probability to introduce raggedness
                        if np.random.uniform() > .5*jag_p:
                            chunk[row, col] +=1
                            
                # next level
                neighbor -= slope
                
        # cap to max_coins
        base[base>max_coin] = [np.random.randint(max_coin+1) for c in base[base>max_coin]]
    
    # save map into txt file
    with open("./maps/{}.txt".format(map_name), "w") as f:
        for line in base:
            f.write(" ".join(map(str, line)) + "\n")


def simulateStep(map_, map_base, input_dict, agent_coords, agent_wealth, agent_gains, dead_agent_id_list, death_coords, step):
    
    # array to store new coin gains in current step
    new_gains_list = np.zeros(input_dict["nAgents"], dtype=float)

    # array to store new wealth in current step
    new_wealth_list = np.zeros(input_dict["nAgents"], dtype=float)

    # array to store new cell coordinates in current step
    new_coords_list = np.zeros(input_dict["nAgents"], dtype=complex)
    
    # list to check uniqueness of new coordinates, unordered
    temp = []
    
    # loop over agents, this can be randomized so that priorities change
    for i in range(input_dict["nAgents"]):
    
        # if agent i is alive
        if i not in dead_agent_id_list:
            scout_dict = lookAround(agent_coords[step, i], map_, input_dict["scoutRange"])
            pref_coords_with_dist, sorted_costs = makePrefList(scout_dict, input_dict["consumptionRate"], agent_coords[step, i])
            
            # preferred step to next cell
            pref_idx = 0  
            
            # move single agent to new cell
            new_coord, new_gain, new_wealth, is_dead = moveAgent(map_, pref_coords_with_dist, agent_wealth[step, i], pref_idx, sorted_costs)
            new_coords_list[i] = new_coord
            temp.append(new_coord)
            new_gains_list[i] = new_gain
            new_wealth_list[i] = new_wealth

            # make sure no two agents step to the same cell; if so choose next cell in preference list for the second agent
            while len(set(temp)) < len(temp):
                pref_idx += 1
                
                # if there is a next best option for agent i; in worst case agent stays in old cell
                if pref_idx < len(pref_coords_with_dist):
                    new_coord, new_gain, new_wealth, is_dead = moveAgent(map_, pref_coords_with_dist, agent_wealth[step, i], pref_idx, sorted_costs)
                    new_coords_list[i] = new_coord
                    temp.pop()
                    temp.append(new_coord)
                    new_gains_list[i] = new_gain
                    new_wealth_list[i] = new_wealth
            
            # if agent i died in current step, save ID and dying coordinates
            if is_dead:
                dead_agent_id_list = np.append(dead_agent_id_list, i)
                death_coords[i] = new_coord
                
        # if agent i died in last step, remove from map
        else:
            new_gains_list[i] = np.nan
            new_wealth_list[i] = np.nan
            new_coords_list[i] = np.nan + np.nan*1j
    
    # append new info from current step and update map
    agent_coords = np.vstack([agent_coords, np.asarray(new_coords_list)])
    agent_gains = np.vstack([agent_gains, new_gains_list])
    print(new_wealth_list)
    print(agent_wealth)
    agent_wealth = np.vstack([agent_wealth, new_wealth_list])
    map_ = updateMap(map_, agent_coords, step, map_base, dead_agent_id_list, refill_rate=input_dict["refillRate"])
    
    # print some logging and the new map
    print('Round {}'.format(step + 1))
    print('Coordinates history:\n{}'.format(agent_coords))
    print('Gains history:\n{}'.format(agent_gains))
    print('Wealth of agents:\n{}'.format(agent_wealth))
    print('Number of living agents: {}'.format(np.count_nonzero([map_ == i for i in [-1, -10, -11]])))
    print('Number and IDs of dead agents: {}, {}'.format(len(dead_agent_id_list), dead_agent_id_list))
    print('Coordinates of death:\n{}'.format(death_coords))
    print("\n")
    print(map_)
    
    # exit simulation if all agents died in current step
    if all(np.isnan(v) for v in agent_wealth[-1,:]):
        print('All agents have died in round {}!'.format(step + 1))
        return map_, 0, agent_coords, dead_agent_id_list, death_coords, agent_gains, agent_wealth
    
    # otherwise plot progress
    else:
        plotWealthDist(agent_wealth[-1,:], step + 1, input_dict["plotFreq"])
        return map_, 1, agent_coords, dead_agent_id_list, death_coords, agent_gains, agent_wealth