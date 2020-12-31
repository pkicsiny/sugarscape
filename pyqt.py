from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import sys

from PIL import Image, ImageQt


import src

class MainWindow(QMainWindow):
    
    
    def checkInputs(self):
        """
        Checks user provided inputs.
        """

        return src.checkInputs(self.input_dict)

    def checkEmpty(self, widget):
        """
        Ensure that user has selected a map.
        :param widget: qt widget item
        """

        if isinstance(widget, QComboBox):
            check = str(widget.currentText())
        else:
            check = str(widget.text())
        if check == "" or check == 'Select a map':
            return 0
        else:
            return 1

    def resetMap(self):
        """
        Reset map.
        """

        temp = np.zeros((self.map_size, self.map_size))
        temp += self.map_base
        return temp

    def startSimulation(self):
        """
        Executes when run button is pressed
        """

        # check if all user inputs are provided except slide bar and reproduction params
        if any(self.checkEmpty(self.input_fields[i]) == 0 for i in range(len(self.input_fields)-3)):
            self.info_msg = 'Enter all inputs!'
            self.paint_step = False
            self.paint_info = True
            self.update()
            
        # if user provided all inputs
        else:
            
            """
            simulation_state:
            0: paused
            1: running
            2: stopped and restarted with same map
            3: fresh start
            """
            
            # initialization
            # if simulatin starts from beginning
            if self.simulation_state >= 2:
                
                # init or clean plot directory
                os.makedirs("plots") if not os.path.isdir("plots") else [os.remove("plots/"+f) for f in os.listdir("plots")]
                
                # if an old simulation has been restarted
                if self.simulation_state == 2:
                    
                    # reset map
                    self.map_ = self.resetMap()
                
                """
                parse user inputs
                """
                self.input_dict = {"mapDim": self.map_size, 
                                   "nAgents": self.input_fields[1].text(),
                                   "maxAge": self.input_fields[2].text(),
                                   "scoutRange": self.input_fields[3].text(),
                                   "consumptionRate": self.input_fields[4].text(),
                                   "refillRate": self.input_fields[5].text(),
                                   "initWealth": self.input_fields[6].text(),
                                   "simSteps": self.input_fields[7].text(),
                                   "reproductionCost": self.input_fields[9].text(),
                                   "plotFreq": self.input_fields[7].text()}
                
                # check user inputs and if ok, run simulation
                if self.checkInputs() == 0:
                    
                    # input fields are disabled during run
                    self.toggleInputFields(True)
                    
                    # simulation is able to run
                    self.can_run = 1
                    
                    """
                    init arrays
                    """
                    if self.reproduction_enabled:
                        
                        # gender: 1=male, 0=female
                        self.gender = np.random.binomial(1, 0.5, size=self.input_dict["nAgents"])
    
                        # attractivity: uniform [0.5, 1]
                        self.attractivity = np.random.uniform(0.5, 1, size=self.input_dict["nAgents"])
    
                        # fertility:
                        # 1: can reproduce (after *maturity_threshold* rounds alive)
                        # 0: cannot reproduce (default, until *maturity_threshold* rounds alive)
                        # -1: pregnant (only females)
                        self.fertility = np.zeros(self.input_dict["nAgents"], dtype=int)
                    else:
                        
                        # when no reproduction no gender
                        self.gender = np.zeros(self.input_dict["nAgents"]) - 1
                    
                    self.agent_coords = src.getCoordinates(self.input_dict)
                    self.agent_gains, self.map_ = src.spawnAgents(self.agent_coords, self.map_, self.input_dict, self.gender)
                    self.agent_wealth = np.zeros_like(self.agent_gains, dtype=float)
                    self.agent_wealth[0, :] = self.agent_gains[0, :] + self.input_dict["initWealth"]
                    self.death_coords = np.zeros(self.input_dict["nAgents"], dtype=complex)*np.nan
                    self.dead_agent_id_list = []
                    
                    # track number of livig agents, births and deaths in a list for plotting
                    self.n_living_agents = [self.input_dict["nAgents"]]
                    self.n_births = [0]
                    self.n_deaths = [0]
                    
                    """
                    iterator
                    """
                    self.it = 0
                    self.step_to_print = 0
                    
                    self.changeMapLayout(self.map_number, 1)
                    self.info_msg = 'Simulation started!'
                    self.paint_step = False
                    self.paint_info = True
                    self.update()
                    self.start_button.setText('Pause')
                    self.start_button.setStyleSheet("background-color: rgba(225,175,20, 0.5)")
                
                    # process initialization
                    qApp.processEvents()
                    time.sleep(1)
                    
                # if user inputs are incorrect
                else:
                    
                    # simulation can not run
                    self.can_run = 0
                    
                    # print info for user to correct inut
                    self.info_msg = self.checkInputs()
                    self.paint_step = False
                    self.paint_info = True
                    self.update()
            
            # if simulation starts from stopped state
            if self.simulation_state != 1 and self.can_run == 1:
                self.info_msg = 'Simulation started!'
                self.paint_step = False
                self.paint_info = True
                self.update()
                
                # enable stop button
                self.stop_button.setDisabled(False)
                
                # change start button text to pause
                self.start_button.setText('Pause')
                self.start_button.setStyleSheet("background-color: rgba(225,175,20, 0.5)")
                
                #simulation is running
                self.running = True
                self.simulation_state = 1
                
                # start measuring runtime
                t0 = time.time()
                
                # plot initial wealth distribution and number of living agents
                src.plotWealthDist(self.agent_wealth[-1,:], 0, self.input_dict["plotFreq"], self.save_plots)
                src.plotLivingAgentCount(self.n_living_agents, self.n_births, self.n_deaths,
                                         0, self.input_dict["plotFreq"], self.save_plots)
                
                
                """
                simulation
                """
                while self.running and self.it < self.input_dict["simSteps"]:
                    
                    # after each iteration print elapsed time
                    t1 = time.time()
                    self.dt = t1 - t0 + self.elapsed_time
                    time_info = str(round(self.dt, 2))
                    num_decimals = len(time_info.split(".")[-1])
                    filler_space = "" if num_decimals == 2 else "0"
                    self.time_widget.setText('Time elapsed: {}{}'.format(time_info, filler_space))
                    
                    # array to store new coin gains in current step
                    new_gains_list = np.zeros(self.input_dict["nAgents"], dtype=float)
                
                    # array to store new wealth in current step
                    new_wealth_list = np.zeros(self.input_dict["nAgents"], dtype=float)
                
                    # array to store new cell coordinates in current step
                    new_coords_list = np.zeros(self.input_dict["nAgents"], dtype=complex)
                    
                    # dead agent counter
                    died_in_current_it = 0
                    
                    # list to check uniqueness of new coordinates, unordered
                    temp = []
                    
                    # list to store spawn coords and wealth of new offsprings
                    if self.reproduction_enabled:
                        offspring_coords = []
                        offspring_wealth = []
                        offspring_abortion = []
                    
                    # loop over agents, agent order is randomized so that priorities change
                    agent_id_list = np.arange(0, self.input_dict["nAgents"], 1)
                    np.random.shuffle(agent_id_list)
                    for i in agent_id_list:
                        
                        # if agent i is alive
                        if i not in self.dead_agent_id_list:
                            
                            """
                            look around
                            """
                            scout_dict = src.lookAround(self.agent_coords[self.it, i],
                                                        self.map_,
                                                        self.input_dict["scoutRange"])
                            
                            """
                            reproduction
                            """
                            if self.reproduction_enabled:
                                self.fertility, offspring_coords, offspring_coords, offspring_abortion = src.reproduce(
                                    i, self.agent_coords[self.it], self.agent_wealth[self.it],
                                    self.gender, self.attractivity, scout_dict, self.input_dict["reproductionCost"],
                                    self.fertility, offspring_coords, offspring_wealth, offspring_abortion)
                                                                     
                            """
                            rank preferred movement choices
                            """
                            pref_coords_with_dist, sorted_costs = src.makePrefList(scout_dict,
                                                                                   self.input_dict["consumptionRate"],
                                                                                   self.agent_coords[self.it, i])
                            
                            # preferred step to next cell
                            pref_idx = 0  
                            
                            """
                            move single agent to new cell
                            """
                            new_coord, new_gain, new_wealth, is_dead = src.moveAgent(self.map_, 
                                                                                     pref_coords_with_dist, 
                                                                                     self.agent_coords[self.it, i],
                                                                                     self.agent_wealth[self.it, i], 
                                                                                     pref_idx, 
                                                                                     sorted_costs)
                            
                            # dies now if reaches max. age (new row is apended to wealth array afterwards)
                            if not is_dead and self.input_dict["maxAge"] != 0 and \
                            np.count_nonzero(~np.isnan(self.agent_wealth[:,i])) == self.input_dict["maxAge"]:
                                new_wealth = np.nan
                                is_dead = 1
                            
                            new_coords_list[i] = new_coord
                            temp.append(new_coord)
                            new_gains_list[i] = new_gain
                            new_wealth_list[i] = new_wealth

                            # make sure no two agents step to the same cell; if so choose next cell in preference list for the second agent
                            while len(set(temp)) < len(temp):
                                pref_idx += 1
                            
                                # if there is a next best option for agent i
                                if pref_idx < len(pref_coords_with_dist):
                                    new_coord, new_gain, new_wealth, is_dead = src.moveAgent(self.map_,
                                                                                             pref_coords_with_dist,
                                                                                             self.agent_coords[self.it, i],
                                                                                             self.agent_wealth[self.it, i],
                                                                                             pref_idx,
                                                                                             sorted_costs)
                                    
                                    # dies now if reaches max. age (new row is apended to wealth array afterwards)
                                    if not is_dead and self.input_dict["maxAge"] != 0 and \
                                    np.count_nonzero(~np.isnan(self.agent_wealth[:,i])) == self.input_dict["maxAge"]:
                                        new_wealth = np.nan
                                        is_dead = 1

                                    new_coords_list[i] = new_coord
                                    temp.pop()
                                    temp.append(new_coord)
                                    new_gains_list[i] = new_gain
                                    new_wealth_list[i] = new_wealth      
                            
                            """
                            offspring abortion
                            """
                            if self.reproduction_enabled:
                                # if mother was pregnant, set fertility back for next round
                                if self.fertility[i] == -1:
                                    self.fertility[i] = 0
                                    
                                    # if mother could move to a new cell, offspring can be born
                                    if new_coords_list[i] != self.agent_coords[self.it, i]:
                                        offspring_abortion[-1] = 0
                                    
                            """
                            agent i died in current step
                            """                       
                            # if agent i died in current step, save ID and dying coordinates
                            if is_dead:
                                self.dead_agent_id_list = np.append(self.dead_agent_id_list, i)
                                self.death_coords[i] = new_coord
                                died_in_current_it += 1
                                
                        # if agent i died in last step, remove from map
                        else:
                            new_gains_list[i] = np.nan
                            new_wealth_list[i] = np.nan
                            new_coords_list[i] = np.nan + np.nan*1j
                    
                    """
                    updates for next iteration
                    """
                    # append new info from current step and update map
                    self.agent_coords = np.vstack([self.agent_coords, new_coords_list])
                    self.agent_gains = np.vstack([self.agent_gains, new_gains_list])
                    self.agent_wealth = np.vstack([self.agent_wealth, new_wealth_list])
                    self.n_living_agents.append(np.count_nonzero(~np.isnan(self.agent_wealth[-1,:])))
                    
                    # track demographics only for plotting
                    self.n_births.append(0)
                    self.n_deaths.append(died_in_current_it)
                    
                    if self.reproduction_enabled:
                        # spawn newborn offsprings
                        for o in range(len(offspring_coords)):
                            
                            # spawn and init new agent only if not aborted
                            if not offspring_abortion[o]:
                                self.input_dict["nAgents"] += 1
                                
                                offspring_coords_list = np.zeros((np.shape(self.agent_coords)[0], 1), dtype=complex)*np.nan
                                offspring_coords_list[-1, 0] = offspring_coords[o]
                                self.agent_coords = np.hstack([self.agent_coords, offspring_coords_list])
                                
                                offspring_gains_list = np.ones((np.shape(self.agent_gains)[0], 1), dtype=int)*np.nan
                                offspring_gains_list[-1, 0] = offspring_wealth[o] + self.input_dict["initWealth"]
    
                                self.agent_gains = np.hstack([self.agent_gains, offspring_gains_list])
                                
                                offspring_wealth_list = np.ones((np.shape(self.agent_wealth)[0], 1), dtype=int)*np.nan
                                offspring_wealth_list[-1, 0] = offspring_wealth[o] + self.input_dict["initWealth"]
    
                                self.agent_wealth = np.hstack([self.agent_wealth, offspring_wealth_list])
                                
                                self.death_coords = np.hstack([self.death_coords, np.zeros(1, dtype=complex)*np.nan])
                                
                                self.n_living_agents[-1] += 1
                        
                                # only for plotting
                                self.n_births[-1] += 1

                                self.gender = np.hstack([self.gender, np.random.binomial(1, 0.5, size=1)])
                                self.attractivity = np.hstack([self.attractivity, np.random.uniform(0.5, 1, size=1)])
                                self.fertility = np.hstack([self.fertility, 0])
                            
                    # map update
                    self.map_, self.map_refill = src.updateMap(self.map_, self.agent_coords, self.it, self.map_base,
                                              self.dead_agent_id_list, self.gender, self.map_refill, 
                                              refill_rate=self.input_dict["refillRate"])
                    
                    if self.reproduction_enabled:
                        # update fertility info for next round
                        for i in range(self.input_dict["nAgents"]):
                            
                            # if agent i is still alive
                            if i not in self.dead_agent_id_list:
                                
                                # if agent i gets mature i.e. older than *maturity_threshold* rounds
                                if self.fertility[i] == 0 and \
                                np.count_nonzero(~np.isnan(self.agent_wealth[:,i])) > self.maturity_threshold:
                                    self.fertility[i] = 1
                            
                            # if agent died, set ferility to 0
                            else:
                                if self.fertility[i] != 0:
                                    self.fertility[i] = 0
                    
                    # update map and status layout
                    self.step_to_print += 1
                    self.changeMapLayout(self.map_number, 1)
                        
                    # process iteration and update interface
                    qApp.processEvents()
                    time.sleep(self.sim_speed)
                    
                    """
                    exit conditions
                    """
                    # exit if map got overpopulated by reproduction
                    if self.n_living_agents[-1] >= self.input_dict["mapDim"]**2:
                        self.handleStop()
                        self.info_msg = 'Maximum number of agents\nreached in round {}!'.format(self.step_to_print)
                        self.paint_step = False
                        self.paint_info = True
                        self.update()
                        self.break_sim = True
                    
                    # exit if all agents died in current step
                    elif all(np.isnan(w) for w in self.agent_wealth[-1,:]):
                        self.handleStop()
                        self.info_msg = 'All agents have died in round {}!'.format(self.step_to_print)
                        self.paint_step = False
                        self.paint_info = True
                        self.update()
                        self.break_sim = True
                    
                    # exit if simulation is finished
                    elif self.it == self.input_dict["simSteps"] - 1:
                        self.handleStop()
                        self.info_msg = 'Simulation with {} steps finished!'.format(self.step_to_print)
                        self.paint_step = False
                        self.paint_info = True
                        self.update()
                        self.break_sim = True
                    
                    # make plots and break
                    if self.break_sim:
                        self.break_sim = False
                        src.plotWealthDist(self.agent_wealth[-1,:], self.step_to_print,
                                           1, self.save_plots)
                        src.plotLivingAgentCount(self.n_living_agents, self.n_births, self.n_deaths,
                                                 self.step_to_print, 1, self.save_plots)
                        break
                    else:
                        src.plotWealthDist(self.agent_wealth[-1,:], self.step_to_print,
                                           self.input_dict["plotFreq"], self.save_plots)
                        src.plotLivingAgentCount(self.n_living_agents, self.n_births, self.n_deaths,
                                                 self.step_to_print, self.input_dict["plotFreq"], self.save_plots)

                    # next iteration
                    self.it += 1
                    
                    
            
            # if simulation is currently running and run button was pressed last
            elif self.simulation_state == 1:
                
                # pause and update run button
                self.start_button.setText('Continue')
                self.start_button.setStyleSheet("background-color: rgba(50,255,50, 0.5)")
                
                # simulation is paused
                self.running = False
                self.simulation_state = 0
                self.elapsed_time = self.dt

    def handleStop(self):
        """
        Executes when simulation is stopped.
        """

        self.running = False
        self.simulation_state = 2
        self.elapsed_time = 0
        self.dt = 0
        self.it = 0
        
        # disable stop button and enable input fields
        self.stop_button.setDisabled(True)
        self.toggleInputFields(False)
        
        # enable run button
        self.start_button.setText('Restart')
        self.start_button.setStyleSheet("background-color: rgba(50,255,50, 0.5)")
    
    def toggleInputFields(self, toggle):
        """
        Enable or disable input fields. All except slide bar.
        :param toggle: bool
        """

        # skip sim speed slide bar and reproduction cost if it is disabled
        last_idx = -1 if self.reproduction_enabled else -2
        for input_field in self.input_fields[:last_idx]:
            input_field.setDisabled(toggle)
            
    def toggleReproduction(self):
        """
        Enable or disable reproduction of agents, depending on the state of the checkbox.
        """
        
        self.reproduction_enabled = self.input_fields[-3].isChecked()
        self.input_fields[-2].setDisabled(not self.reproduction_enabled)

    def changeSimSpeed(self):
        """
        Change simulation speed based on slide bar value.
        """

        # slide bar is the last input field
        self.input_fields[-1].setToolTip(str(1+self.input_fields[-1].value()))
        sim_speed = int(self.input_fields[-1].value())
        self.sim_speed = self.sleep_times[sim_speed]

    def initMapLayout(self):
        """
        Initialize map layout with a welcome message.
        Executes when app is started up or the "select map" option is picked from the dropdown.
        """
        
        # set default map size to prevent paintevent from drawing
        self.map_size = 0
        self.paint_welcome = True
        self.paint_cells = False 
        self.paint_step = False
        self.paint_info = False
        self.update()

    def changeMapLayout(self, mapID, draw = False):
        """
        Executes when user selects another map from the dropdown.
        :param mapID: int, number of map.
        :param draw: bool, true during simulation
        :param print_step: bool, print current iteration number
        """

        # draw selected map on GUI.
        if not draw:
            
            # initialize run button and set simulation state to "fresh start"
            self.start_button.setText('Start')
            self.simulation_state = 3
        
        if mapID == 0 and not draw:
            self.initMapLayout()
        else:
            
            # load map (dropdown indices start from 1
            if not draw:              
                self.map_base = np.loadtxt("./maps/{}".format(self.map_list[int(mapID)-1]))
                self.map_ = np.loadtxt("./maps/{}".format(self.map_list[int(mapID)-1]))
                self.map_number = mapID
                self.map_size = len(self.map_base)
                
                # map for coin regrowth tracing
                self.map_refill = np.zeros_like(self.map_, dtype=int)
                
                # do not print step number when new map is loaded
                self.paint_step = False
            else:
                self.paint_step = True
                
            t0 = time.time()
                            
            # enable drawing map and call paint event
            self.paint_welcome = False 
            self.paint_cells = True 
            self.paint_info = False
            self.update()


    def paintEvent(self, event):
        """
        Builtin draw method.
        """
        
        # init qpainter
        qpainter = QPainter()
        qpainter.begin(self)
        
        # clean canvas
        qpainter.eraseRect(0, 0, self.window_width, self.window_height)
        
        # if new map has to be drawn
        if self.paint_welcome:
            self.drawWelcome(event, qpainter)
        
        # only possible if user has loaded a map
        if self.map_size > 0:
            if self.paint_cells:
                self.drawWelcome(event, qpainter)
                self.drawMap(qpainter)  
                
            if self.paint_step:
                self.drawStep(qpainter)
        
        # paint simulation info
        if self.paint_info:
            self.drawInfo(qpainter)
            
        qpainter.end()

    def drawWelcome(self, event, qpainter):
        """
        Draw welcome text that appears upon starting.
        :param qpainter: QPainter object, handle for drawing
        """

        qpainter.setPen(QColor(0, 0, 0))
        qpainter.setBrush(Qt.transparent)
        qpainter.setFont(QFont('Times', 20))
        
        # draw main bounding rectangle
        qpainter.drawRect(self.main_rect)
        qpainter.drawText(self.main_rect, Qt.AlignCenter, 'Welcome!\n Select a map!')
        qpainter.drawRect(self.status_rect)
            
    def drawStep(self, qpainter):
        """
        Draw step iterator.
        :param qpainter: QPainter object, handle for drawing
        """

        qpainter.setPen(QColor(0, 0, 0))
        qpainter.setBrush(Qt.transparent)
        qpainter.setFont(QFont('Times', 10))
        
        # draw main bounding rectangle
        qpainter.drawRect(self.status_rect)
        if self.step_to_print == 0:
            qpainter.drawText(self.status_rect, Qt.AlignCenter, 'Simulation started!')
        else:
            qpainter.drawText(self.status_rect, Qt.AlignCenter, 'Round ' + str(self.step_to_print))
            
    def drawInfo(self, qpainter):
        """
        Draw simulation information.
        :param qpainter: QPainter object, handle for drawing
        """

        qpainter.setPen(QColor(0, 0, 0))
        qpainter.setBrush(Qt.transparent)
        qpainter.setFont(QFont('Times', 10))
        
        # draw main bounding rectangle
        qpainter.drawRect(self.status_rect)
        qpainter.drawText(self.status_rect, Qt.AlignCenter, self.info_msg)
    
    def drawMap(self, qpainter):
        """
        Draw map with rectangles on canvas.
        :param qpainter: QPainter object, handle for drawing

        """     

        # plus half margin to be aligned
        cells_left = np.linspace(self.map_grid_left_offset+self.map_grid_margin,
                                 self.map_grid_left_offset+self.map_grid_margin+self.map_grid_dim_tight,
                                                           self.map_size+1)[:-1]
        cells_top = np.linspace(self.map_grid_top_offset+self.map_grid_margin,
                                self.map_grid_top_offset+self.map_grid_margin+self.map_grid_dim_tight,
                                                            self.map_size+1)[:-1]
        cell_size = int(self.map_grid_dim_tight/(self.map_size)) 
        
        #draw cells one by one
        for row in range(self.map_size):
            for col in range(self.map_size):
                
                # key is an int, value is a color code
                for key, value in self.map_colors.items():
                    
                    # draw a new cell
                    if self.map_[row][col] == key:
                        qpainter.setBrush(value)
                        qpainter.drawRect(cells_left[col], cells_top[row], cell_size, cell_size)
                        
                        # gendered cells
                        if key in [-10, -20, -30, -11, -21, -31]:
                            quarter_cell = .25*cell_size+1
                            half_cell = .5*cell_size
                            gender_color = "#0000ff" if key in [-11, -21, -31] else "#ff1493"
                            qpainter.setBrush(QBrush(QColor(gender_color)))
                            qpainter.setPen(Qt.transparent)
                            qpainter.drawRect(cells_left[col]+quarter_cell, cells_top[row]+quarter_cell, half_cell, half_cell)
                            qpainter.setPen(QColor(0, 0, 0))

    
    def __init__(self):
        """
        Constructor.
        """

        super().__init__()
        
        
        """
        gui setup
        """
        # header
        self.setWindowTitle("Sugarscape")
        
        # footer
        self.statusBar().showMessage('Epstein & Axtell Sugarscape')
        self.time_widget = QLabel()
        self.statusBar().addPermanentWidget(self.time_widget)
        
        # main window focusing
        self.app_focused = True

        # flags for event draw
        self.paint_cells = False
        self.paint_welcome = False
        self.paint_step = False
        self.paint_info = False
        
        # fix app window size
        self.window_width = 600
        self.window_height = 400
        self.map_layout_width = int(.5*self.window_width)
        self.setFixedSize(self.window_width, self.window_height)
        
        # define map layout details
        self.map_grid_top_offset = 50 # space for two lines
        self.map_grid_dim = self.map_layout_width if self.window_height >= self.map_layout_width \
                                                  else int(self.window_height-self.map_grid_top_offset)
        self.map_grid_left_offset = 0 if self.window_height >= self.map_layout_width \
                                      else int(.5*(self.map_layout_width-self.map_grid_dim))
        self.map_grid_dim_tight = int(.9*self.map_grid_dim)
        self.map_grid_margin = int(.05*self.map_grid_dim) 
        
        # initialize map layout rectangles
        self.main_rect = QRect(self.map_grid_left_offset+self.map_grid_margin,
                          self.map_grid_top_offset+self.map_grid_margin,
                          self.map_grid_dim_tight, self.map_grid_dim_tight)
        
        self.status_rect = QRect(self.map_grid_left_offset+self.map_grid_margin,
                          self.map_grid_margin,
                          self.map_grid_dim_tight, 50 - self.map_grid_margin)
        
        """
        global variables
        """
        # simulation state initialization
        self.running = False
        self.simulation_state = 3
        self.elapsed_time = 0
        self.dt = 0
        
        # iterator and auxiliary step counter for printing
        self.it = 0
        self.step_to_print = 0
        
        # break simulation
        self.break_sim = False
        
        """
        global parameters
        """
        
        # info message
        self.info_msg = ""
        
        # plot saving
        self.save_plots = True
        
        # reproduction
        self.reproduction_enabled = False
        
        # maturity threshold
        self.maturity_threshold = 20
        
        # simulation speed: sleep time between iterations
        self.sleep_times = [0.5, 0.4, 0.3, 0.2, 0.1, 0.075, 0.05, 0.025, 0.01, 0]
        self.sim_speed = self.sleep_times[6]
        
        # coin - color correspondence
        self.map_colors = {-3: QBrush(QColor("#09810c")), # spawn neutral
                           -30: QBrush(QColor("#09810c")), # spawn female
                           -31: QBrush(QColor("#09810c")), # spawn male
                           -2: QBrush(QColor("#b22222")), # dead neutral
                           -20: QBrush(QColor("#b22222")), # dead female
                           -21: QBrush(QColor("#b22222")), # dead male
                           -1: QBrush(QColor("#000000")), # living neutral
                           -10: QBrush(QColor("#000000")), # living female
                           -11: QBrush(QColor("#000000")), # living male
                           0: QBrush(QColor("#ffffe0")),
                           1: QBrush(QColor("#eee8aa")),
                           2: QBrush(QColor("#eedd82")),
                           3: QBrush(QColor("#ffd700")),
                           4: QBrush(QColor("#daa520")),
                           5: QBrush(QColor("#b8860b"))}
        
        """
        build window layout
        """
        # 2 horizontal layouts: map and control panel 
        #central widget of window
        self.widget = QWidget()
        self.window_layout = QHBoxLayout()
        self.widget.setLayout(self.window_layout)     
        self.setCentralWidget(self.widget)

        # init map_layout->map_widget->draw_layout (left side)
        self.map_layout = QVBoxLayout()
        
        self.step_layout_widget = QLabel()
        self.map_widget = QWidget()
        self.map_widget.setFixedSize(self.map_layout_width, self.window_height)
        
        self.map_layout.addWidget(self.step_layout_widget)
        self.map_layout.addWidget(self.map_widget)
        self.draw_layout = QVBoxLayout()
        self.map_widget.setLayout(self.draw_layout)
        self.initMapLayout()

        # init gui_layout (right side)
        self.gui_layout = QVBoxLayout()
        
        self.form_layout = QFormLayout()
        self.form_layout.setLabelAlignment(Qt.AlignRight)
        self.control_layout = QHBoxLayout()
        
        self.gui_layout.addLayout(self.form_layout, 1000)
        self.gui_layout.addLayout(self.control_layout, 1000)

        
        # init gui_layout->form_layout->inputs
        # input names
        input_names = {'Map number': "Select a map from the dropdown list.\n"
                                      "Maps are stored in txt files as space separated 2D matrices.",
                     'Number of agents': "Number of agents to simulate.",
                     'Maximum age': "Agents die after this many steps. If 0, agents won't die of old age.",
                     'Vision distance': "Range of vision of agents on the map.\n"
                                         "It tells how many cells they can see in all directions.",
                     'Consumption rate': "Rate of consuming coins.\n"
                                          "Cost of moving to a neighboring cell or staying in the current cell.\n"
                                          "Scales with distance.\n"
                                          "No coin loss if set to 0.",
                     'Refill rate': "Rate of refilling map with coins.\n"
                                     "Each cell fills up until its original value.\n"
                                     "If set to N, cell values increase by 1 after every N rounds.",
                     'Initial wealth': "Initial wealth of each agent upon spawn.\n"
                                        "If set to 0 agents start without any coins.",
                     'Simulation steps': "Number of simulation rounds (=iterations).",
                     'Enable reproduction': "Agents can mate after having lived\n"
                                               "for 20 steps and bear offsprings onto the map.\n"
                                               "Each agent has a gender either male or female and an\n"
                                               "attractivity score randomly assigned between 0.5 and 1.\n"
                                               "If two agents of the opposite sex find each other attractive\n"
                                               "the female gets pregnant and spawns a new agent on its current\n"
                                               "cell upon moving.",
                     'Reproduction cost': "In order to reproduce, the male and female agents\n"
                                          "have to have this amount of coins together.",
                     'Simulation speed': "1: slowest\n"
                                          "10: fastest"}
        
        # input fields
        self.input_fields = []
        self.input_texts = []
        for name, tooltip in input_names.items():
            input_text = QLabel(name)
            input_text.setToolTip(tooltip)
            if name == "Map number":
                input_line = QComboBox()
                input_line.addItem('Select a map')
                self.map_list = os.listdir('./maps')
                input_line.addItems(self.map_list)
                input_line.currentIndexChanged.connect(self.changeMapLayout)
                input_line.setFixedSize(120, 20)
            elif name == "Simulation speed":
                input_line = QSlider(Qt.Horizontal)
                input_line.setMinimum(0)
                input_line.setMaximum(9)
                input_line.setValue(6)
                input_line.setTickPosition(QSlider.TicksBelow)
                input_line.setTickInterval(1)
                input_line.setFixedWidth(120)
                input_line.valueChanged.connect(self.changeSimSpeed)
            elif name == "Enable reproduction":
                input_line = QCheckBox(input_text)
                input_line.setChecked(False)
                input_line.stateChanged.connect(self.toggleReproduction)
            else:
                input_line = QLineEdit(input_text)
                input_line.setText("0")
                input_line.setFixedSize(120, 20)
                input_line.setAlignment(Qt.AlignRight)
            if name == "Reproduction cost":
                input_line.setText("0")
                input_line.setDisabled(True)
            self.input_texts.append(input_text)
            self.input_fields.append(input_line)
            self.form_layout.addRow(self.input_texts[-1], self.input_fields[-1])

        # init gui_layout->control
        self.start_button = QPushButton('Start')
        self.start_button.clicked.connect(self.startSimulation)
        self.start_button.setFixedSize(100, 50)
        self.start_button.setFont(QFont("Times", 14, QFont.Bold))
        self.start_button.setStyleSheet("background-color: rgba(50,255,50, 0.5)")
        self.stop_button = QPushButton('Stop')
        self.stop_button.setDisabled(True)
        self.stop_button.clicked.connect(self.handleStop)
        self.stop_button.setFixedSize(100,50)
        self.stop_button.setFont(QFont("Times", 14, QFont.Bold))
        self.stop_button.setStyleSheet("background-color: rgba(255,50,50, 0.5)")
        
        self.control_layout.addWidget(self.start_button)
        self.control_layout.addWidget(self.stop_button)
        
        
        # add layouts to main widget
        self.window_layout.addLayout(self.map_layout)
        self.window_layout.addLayout(self.gui_layout)
