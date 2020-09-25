# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 16:14:08 2020

@author: c3273262
"""
import numpy as np
import pandas as pd
import pickle
import json
import os
from datetime import datetime, timedelta
from copy import deepcopy
from sys import argv, getsizeof
from matplotlib import pyplot
from matplotlib import colors as mcolors
from matplotlib import patches as mpatches
from pathlib import Path


#===============================================================================================================    
# LIBRARY GLOBAL PARAMETERS
#===============================================================================================================
#Default name for some files accessed in many routines
fn_ = {
    'setup': 'anpanta.dat',
    'bounds': 'contar.dat',
    'init': 'inicial.dat',
    'param': 'param.dat',
    'domain': 'vincul.dat',
    'h': ['depths_','.txt'],
    'v': ['veloc_','.txt'],
    'Q': ['flows_','.txt'],
    'tiles': 'special_tiles.dat',
    'profiles': 'profiles.dat',
    'exec': 'acc_hydro.exe'
}
#===============================================================================================================



#===============================================================================================================
# LIBRARY CLASSES
#===============================================================================================================
class EGMframework(object):
    """ Object to manage information and procedures to run a
    Eco-GeoMorphological simulation. """
    
    #-----------------------------------------------------------------------------------------------------------
    def __init__(self, title=None):
        self.title = title
        self._timeStart = [datetime.now()]
        self._timeFinish = []
        
        # Default parameters
        self.w_new = 1.0    #new vegetation assumes 100% of the new roughness
        self.Dcoef = 0.0    #hillslope diffusion is not applied
        self.cycles = 86400 #time-length (in seconds) of max-periods' intervals
        self.hfd_start = 0  #high-frequency data start (ignore data in the seconds before this on EGM computations)
        self.hfd_last = 0   #hfd last seconds that will be ignored
        self.rule = 'SE Australia' #rule to be followed in the vegetation and accretion models
        self.flood_depth_threshold = 0.14 #depth limit to compute hydroperiod
        self.resume = False
        self.maxsteps = None
        self.first_column_as_index = False    #boolean to decide if input series csv file has an index column
    #-----------------------------------------------------------------------------------------------------------
    
    #-----------------------------------------------------------------------------------------------------------
    def add_settings_attributes(self, attributes_dict):
        #Parsing dictionary items to self-object properties
        for key, value in attributes_dict.items():
            
            if key in ['Dt_EGM','Dt_HST', 'Dt_inputs', 'nrows', 'ncols', 'maxsteps']:
                #attributes with integer value
                setattr(self, key, int(value))
                
            elif key in ['Dcoef', 'w_new', 'Ucd', 'ws']:
                #attributes with float value
                setattr(self, key, float(value))
                
            elif key in ['resume', 'first_column_as_index']:
                #attributes with boolean value
                if value.lower() in ["y", "yes", "t", "true", "on", "1"]:
                    setattr(self, key, True)
                elif value.lower() in ["n", "no", "f", "false", "off", "0"]:
                    setattr(self, key, False)
                else:
                    raise ValueError(str('\n Option for %s is not valid\n' % key))
            
            elif key == 'roughness':
                #special treatment for roughness values
                if not hasattr(self, 'roughness'):
                    setattr(self, 'roughness', {})                
                for vegetation_type, manning in value.items():
                    self.roughness[int(vegetation_type)] = float(manning)
                
            else:
                #attributes that are string
                setattr(self, key, value)
    #-----------------------------------------------------------------------------------------------------------
    
    #-----------------------------------------------------------------------------------------------------------
    def finalize(self, filename='framework_data.pickle'):
        self._timeFinish.append(datetime.now())
        
        if filename.endswith('pickle') or filename.endswith('pkl'):
            # Saving all EGMframework instance in a pickle file
            with open(filename, 'wb') as file:
                pickle.dump(self, file)
        
        if filename.endswith('csv'):
            # Saving domain properties and EGM results in a CSV file
            file = open(filename, 'w')
            file.write('Stage,Cell,x (m),y (m),z (m),dx (m),dy (m),chndep (m)' +
                ',Hydroperiod (%),Mean Depth Below High Tide (m),Average SSC (g/m3),Vegetation code,' +
                'Accretion (m),Elevation Gain(m)\n')
            
            for i in range(self.Nstages):
                
                if i == 0:
                    z = self.z0
                else:
                    z = self.z0 + self.EG[i-1,:]
                
                for j in range(self.NoC):
                    file.write('%s,%i,%.1f,%.1f' % (self.stages[i], self.number[j], self.x[j], self.y[j]))
                    file.write(',%.4f,%.1f,%.1f,%.1f' % (round(z[i],4), self.dx[j], self.dy[j], self.chndep[j]))
                    file.write(',%.2f,%.4f,' % (round(self.H[i,j],2), round(self.D[i,j],4)))
                    file.write('%.1f,%i,%.4f' % (round(self.C[i,j],1), self.V[i,j], round(self.A[i,j],4)))
                    file.write(',%.4f\n' % (round(self.EG[i,j],4)))
            
            file.close()
    #-----------------------------------------------------------------------------------------------------------
    
    #-----------------------------------------------------------------------------------------------------------
    def initialize(self, options_file, change_to_folder=None):
        print('Initializing EGMframework instance:')
        
        # Dealing with inline arguments
        #-------------------------------------------------------------------------------------------------------
        #Reading file with EGM simulation settings and adding them as attributes
        opts = egm_settings_read(options_file)        
        self.add_settings_attributes(opts)
        print('- %i settings definitions from settings file' % (len(opts)))
        
        #Change current folder to where the hydrodynamic files are
        if change_to_folder != None:
            os.chdir(change_to_folder)
            print('- Current folder changed to:', os.getcwd())
        
        # Defining run title if not especified (read from first line of anpanta.dat )
        if self.title is None:
            with open(fn_['setup'], 'r') as file:
                self.title = file.readline()[1:-2]
        #-------------------------------------------------------------------------------------------------------
        
        # Resuming an interrupted framework run
        #-------------------------------------------------------------------------------------------------------
        if self.resume:
            
            #Load the EGMframework object from pickle file in the current folder
            with open('framework_data.pickle', 'rb') as file:
                previous = pickle.load(file)
                self.__dict__.update(previous.__dict__)
                self._timeStart.append(datetime.now())
            
            #Overwrite previous setup options with the current ones from setup file
            self.add_settings_attributes(self, opts)
            
            print('- Resumed previous framework run.')
            return
        #-------------------------------------------------------------------------------------------------------
        
        # Domain-related information
        #-------------------------------------------------------------------------------------------------------
        #Domain info from vincul.dat
        aux = domain_read()
        for key in aux.keys():
            setattr(self, key, aux[key])
        print('- Domain properties: okay')
        
        #Parameters info from param.dat
        aux = parameters_read()
        for key in aux.keys():
            setattr(self, key, aux[key])
        print('- Parameters values: okay')
        
        #HxT boundary conditions
        aux = boundary_depths_read()
        self.loc_HT = aux['loc_HT']
        print('- %i boundaries of HxT type' % (len(self.loc_HT)))
        
        #Special cells (tide-input, river-like)
        if hasattr(self, 'special_tiles_filename'):
            fn_['tiles'] = self.special_tiles_filename
        aux = special_tiles_read()
        setattr(self, 'tiles', aux)
        
        #Define input cells for Sediment Transport model
        self.input_cells = np.array([False for i in range(self.NoC)])
        if 'TIDE' in self.tiles.keys():
            self.input_cells[self.tiles['TIDE']['cells']] = True
        if 'INPUT_CELLS' in self.tiles.keys():
            self.input_cells[self.tiles['INPUT_CELLS']['cells']] = True
        
        #Define cells to be excluded from EGM models
        self.exclude = np.array([False for i in range(self.NoC)])
        for tile in self.tiles.keys():
            self.exclude[self.tiles[tile]['cells']] = True
        print('- Special Tiles: okay')
        
        #Relationship among cells/links (nbcells, links, signal, same_x, same_y)
        aux = relationships(self)
        for key in aux.keys():
            setattr(self, key, aux[key])
        print('- Relationships between cells and links: okay')
        
        #Profiles for application of Hillslope Diffusion model
        if hasattr(self, 'hillslope_profiles_filename'):
            fn_['profiles'] = self.hillslope_profiles_filename
        if self.Dcoef > 0.0:
            self.profiles, self.delta_x = hillslope_profiles_read()
            print('- %i profiles for hillslope diffusion model' % (len(self.profiles)))
        #-------------------------------------------------------------------------------------------------------
        
        # Dealing with input series
        #-------------------------------------------------------------------------------------------------------
        if hasattr(self, 'input_levels_file'):
            
            #Input water level and sediment concentration files as pandas.DataFrame
            self.levels = table_series_read(self.input_levels_file, self.first_column_as_index)
            self.sedcon = table_series_read(self.input_sediment_file, self.first_column_as_index)
            columns = list(self.levels.columns)
            
            #Input series time-step (delta time) if not defined yet
            if (not hasattr(self, 'Dt_inputs')) and (self.first_column_as_index):
                self.Dt_inputs = self.levels.index[1] - self.levels.index[0]
                print('- Inputs time-step set as %i s' % (int(self.Dt_inputs)))
            
            #Creating attributes for first and last stage of EGM simulation if it was not provided
            if not hasattr(self, 'start'):
                self.start = columns[0]
            if not hasattr(self, 'end'):
                self.end = columns[-1]
                        
            #Creating attributes to store the stage names/ids in the same order found in the input file
            i0 = columns.index(self.start)
            iN = columns.index(self.end)
            self.stages = columns[i0:iN+1]
            self.Nstages = len(self.stages)
            print('- %i EGM stages from %s to %s (Inputs series: okay)' % (self.Nstages, self.start, self.end))
            
        else:
            self.Nstages = 1
            self.periods = [None]
            print('- Any EGM stage defined')
        #-------------------------------------------------------------------------------------------------------
        
        # if not provided, delta time of hydrodynamic outputs will be taken from initial conditions file
        if not hasattr(self, 'Dt_HST'):
            self.Dt_HST = initial_condition_read(properties='Dt_output')
            print('- HST output time-step set to %.1f s' % self.Dt_HST)
        
        # if not provided, get delta time of input series from HxT boundary conditions file
        if not hasattr(self, 'Dt_inputs'):
            self.Dt_inputs = boundary_depths_read(properties='Dt_HT')
            print('- Inputs series time-step set to %.1f s' % self.Dt_inputs)
        
        # Identifing intervals to obtain maximum depths        
        if hasattr(self, 'levels'):
            self.periods = []
            for i in range(self.Nstages):
                aux = self.split_intervals(i)    #this functions demands self.Dt_HST
                self.periods.append(aux[:])
                
        # Variables' matrices
        #-------------------------------------------------------------------------------------------------------
        shp = (self.Nstages, self.NoC)
        self.H = np.zeros(shp, dtype=float) * np.nan    #hydroperiod (%)
        self.D = np.zeros(shp, dtype=float) * np.nan    #mean depth below high tide (m)
        self.V = np.zeros(shp, dtype=int) - 1           #vegetation type
        self.C = np.zeros(shp, dtype=float) * np.nan    #step-average sed. conc. (g/m3)
        self.A = np.zeros(shp, dtype=float) * np.nan    #accretion (m)
        self.EG = np.zeros(shp, dtype=float) * np.nan   #elevation gain (m)
        #-------------------------------------------------------------------------------------------------------
        
        #Number of rows and columns in the domain
        #-------------------------------------------------------------------------------------------------------
        if not hasattr(self, 'nrows'):
            aux = np.unique(self.y)
            self.nrows = aux.size
        
        if not hasattr(self, 'ncols'):
            aux = np.unique(self.x)
            self.ncols = aux.size
        #-------------------------------------------------------------------------------------------------------
        
        #Final assignments
        #-------------------------------------------------------------------------------------------------------
        if hasattr(self, 'ws'):
            if self.ws > 0:
                self.ws *= -1
                print('- Settling velocity should be negative. Changed to', self.ws)
                
        self.current = 0
        self.z0 = self.z.copy()
        self.z0[self.chndep > 0] += self.chndep[self.chndep > 0]
        print('Framework initialisation complete.\n')
    #-----------------------------------------------------------------------------------------------------------
    
    #-----------------------------------------------------------------------------------------------------------
    def initialize_hydrodynamic(self, stage=None, veg_stix=None, hfd=None):
        '''
        This method carries on the following tasks:
        1.  Update domain's bottom elevation with current data in egm.z and egm.chndep attributes
        2.  Update surface-roughness parameters given the EGM 'veg_stix', which means, use the vegetation saved
            for this stage index. If this argument is not provided, the parameters are not updated
        3.  Update HxT boundary conditions using the input water-level series at 'stage'. If stage is not
            provided, it takes self.stages[self.current]. The time-step for the input series should be stored
            in the self.Dt_inputs property. If this attribute does not exist, it is not updated.
        4.  Update runtime/initial condition file. The final time of the hydrodynamic simulation is computed
            here to match the same extension of the input series. The time-step for the hydrodynamic output
            files should be stored in the self.Dt_HST attribute. If this one does not exist, it will try to get
            from the hfd.Dt_output (highFreqData object) attribute. If hfd was not provided, then this property
            is not updated. Lastly, the initial water depth comes from hfd.h[-1,:], i.e. the water depths at the
            last time-step. Again, if hfd is not provided, it does not change the initial water levels as well.
        '''
        # Update bottom level (and channel depth at river-type cells)
        domain_update(z=self.z, chndep=self.chndep)
        
        #Update surface roughness
        if veg_stix is not None:
            self.update_parameters(stix=veg_stix)
            parameters_update(params=self.params)
        
        #Update HxT boundary conditions
        if stage is None:
            stage = self.stages[self.current]
        
        if hasattr(self, 'Dt_inputs'):
            boundary_depths_update(levels=self.levels[stage].dropna(), elevs=self.z, Dt_HT=self.Dt_inputs)
        else:
            boundary_depths_update(levels=self.levels[stage].dropna(), elevs=self.z)
        
        #Reviewing final time and output's time step
        n = len(self.levels[stage].dropna()) - 1    #input level series starts at t=0
        ft, delta = initial_condition_read(properties=['t_final', 'Dt_output'])        
        if hasattr(self, 'Dt_HST'):
            delta = self.Dt_HST
        else:
            if hfd is not None and hasattr(hfd, 'Dt_output'):
                delta = hfd.Dt_output
        ft[-1] = n*delta
        
        #Update runtime/initial conditions
        if hfd is not None:
            initial_condition_update(t_final=ft, Dt_output=delta, h0=hfd.h[-1,:])
        else:
            initial_condition_update(t_final=ft, Dt_output=delta)
        
        #Finish
        return True
    #-----------------------------------------------------------------------------------------------------------
    
    #-----------------------------------------------------------------------------------------------------------
    def split_intervals(self, stage_index):
        nd = len(self.levels.iloc[:,stage_index].dropna()) - 1    #subtract one because the boundary levels
        #series is always one record longer than the number of time steps recorded by the hydrodynamic model
        sequences = define_periods(ndata = nd, record_step = self.Dt_HST, single_duration = self.cycles,
                                   skip_first = self.hfd_start, skip_last = self.hfd_last)
        return sequences
    #-----------------------------------------------------------------------------------------------------------
    
    #-----------------------------------------------------------------------------------------------------------
    def update_accretion(self, stix):
        act = accretion(D = self.D[stix], V = self.V[stix], C = self.C[stix], rule = self.rule)
        return act * self.Dt_EGM
    #-----------------------------------------------------------------------------------------------------------
    
    #-----------------------------------------------------------------------------------------------------------
    def update_bathtub(self, stage, returnType=None):
        hb = bathtub_depths(wl = np.array(self.levels[stage]), elev = self.z)
        
        if returnType is None:
            return hb
        
        elif returnType == 'highFreqData':
            data = highFreqData(Ndata = hb.shape[0] - 1, NoC = hb.shape[1], timestep = self.Dt_inputs,
                                h = hb[1:,:])
            data.h0 = hb[0,:]
            return data
        
        else:
            raise ValueError('\n Sorry! Returning bathtub outputs as' + str(returnType) +
                             'is not implemented!\n')
    #-----------------------------------------------------------------------------------------------------------
    
    #-----------------------------------------------------------------------------------------------------------
    def update_diffusion(self):
        ''' Apply the Hillslope Diffusion model for self.Dt_EGM years. It uses the attribute self.z0 as minimum
        elevation ever. This attribute should be the bottom elevation at land-type cells, and the bank elevation
        at river-type cells. '''
        zin = self.z.copy()    #take current elevation
        zin[self.chndep > 0] += self.chndep[self.chndep > 0]    #set elevation in river-cells as bank-elevation
        
        for i in range(self.Dt_EGM):
            zout = hillslope_diffusion(elev=zin, Dcoef=self.Dcoef, lines=self.profiles, dx=self.delta_x)
            fix = zout < self.z0        #cells where new elevation became lower than the initial elevation
            zout[fix] = self.z0[fix]    #don't allow go lower than z0
            zin[:] = zout               #copy data for next iteration/year
        
        self.z[self.chndep == 0] = zout[self.chndep == 0]       #update bottom elevation of land-type cells
        self.chndep[self.chndep > 0] = zout[self.chndep > 0]    #update channel depth of river-type cells
        return zout - self.z0
    #-----------------------------------------------------------------------------------------------------------
    
    #-----------------------------------------------------------------------------------------------------------
    def update_hydrodynamic(self, returnType=None):
        """ Run the Hydrodynamic simulation. If self.maxsteps was not set up (still is None), the model will run
        at once. Otherwise, a procedure to break the simulation in multiple intervals will be used (to reduce
        the size of output txt files)
            If returnType is None, it will return a boolean True. Otherwise, returns the outcomes of the
        hydrodynamic simulation in a highFreqData class object.
        """
        # Deciding about the returning type
        if returnType is None:
            return_outcome = False
        elif returnType == 'highFreqData':
            return_outcome = True
        else:
            raise ValueError('\n Sorry! Returning hydrodynamic outputs as' + str(returnType) +
                             'is not implemented!\n')
            
        if self.maxsteps is None:
            print('\n Running hydrodynamic model: full length ...')
            x = hydrodynamic_model(return_outcome=return_outcome)
            
        else:
            print('\n SORRY! NOT IMPLEMENTED YET!\n')
            exit()
#            print('\n Running hydrodynamic model: multiple intervals ...')
#            x = hydrodynamic_model_mult(maxsteps=self.maxsteps,
#                                        return_outcome=return_hfd)
        return x
    #-----------------------------------------------------------------------------------------------------------
        
    #-----------------------------------------------------------------------------------------------------------
    def update_linear_sediment(self, stix):
        Cmax = self.sedcon[self.stages[stix]].mean()
        Cavg = bathtub_ssc(D = self.D[stix,:], Cmax = Cmax)
        return Cavg
    #-----------------------------------------------------------------------------------------------------------
    
    #-----------------------------------------------------------------------------------------------------------
    def update_parameters(self, stix):
        """ Update the Manning's roughness coefficient in the domain links and other parameters related to
        changes in surface elevation. """
        
        #Creating array with Manning's roughness coefficient on each cell
        #based on vegetation type
        mrc = np.zeros(self.NoC) * np.nan
        
        for area in self.tiles.keys():
            mrc[self.tiles[area]['cells']] = self.tiles[area]['roughness']
        
        for i in range(self.NoC):
            if not self.exclude[i]:
                mrc[i] = self.roughness[self.V[stix,i]]
        
        #Updating parameters
        for i in range(self.NoL):
        
            #Getting the roughness in cell1 and cell2 of the current link
            c1, c2 = self.cell1[i] - 1, self.cell2[i] - 1
            n_new = 0.5 * (mrc[c1] + mrc[c2])
            
            if self.lktype[i] == 61:
                #Land-Land link: update roughness between cells
                self.params[i,2] = self.w_new * n_new + (1 - self.w_new) * self.params[i,2]
                
            elif self.lktype[i] == 6:
                #River-Land link: update roughness between cells and elevation
                #of land-phase
                self.params[i,2] = self.w_new * n_new + (1 - self.w_new) * self.params[i,2]
            
                if self.chndep[c1] > 0.0:    # first cell is the river
                    self.params[i,3] = self.chndep[c1] + self.z[c1]
                    self.params[i,4] = self.z[c2]
                    
                else:    # first cell is land type (second is river type)                
                    self.params[i,3] = self.z[c1]
                    self.params[i,4] = self.chndep[c2] + self.z[c2]
            
            elif self.lktype[i] == 0:
                #River-River link: update land-phase roughness and average channel depth
                self.params[i,3] = self.w_new * n_new + (1 - self.w_new) * self.params[i,3]
                self.params[i,6] = 0.5 * (self.chndep[c1] + self.chndep[c2])
            
            else:
                #Other link types: No changes at all
                pass
    #-----------------------------------------------------------------------------------------------------------  

    #-----------------------------------------------------------------------------------------------------------
    def update_sediment_transport(self, stage, hfd=None):
            
        if hfd is None:    #Loading hydrodynamic outputs to a highFreqData object if hfd is not provided
            hfd = highFreqData()
            hfd.load_native_outputs()
        
        if not hasattr(hfd, 'h0'):    #Loading initial depths if hfd don't have it
            hfd.load_h0()
            
        if not hasattr(hfd, 'ssc0'):    #initial condition for ssc = 0 g/m3, if hfd don't have it
            hfd.ssc0 = np.zeros(self.NoC)
        
        #calling the sediment transport model and placing its outputs in hfd.ssc
        hfd.ssc = sediment_transport_model(fco=self, sscIn=np.array(self.sedcon[stage].dropna()),
            ssc0=hfd.ssc0, h=hfd.h, v=hfd.v, Q=hfd.Q, h0=hfd.h0)
        
        return hfd
    #-----------------------------------------------------------------------------------------------------------
    
    #-----------------------------------------------------------------------------------------------------------
    def update_tidal_indexes(self, stix, depths):
        return tidal_indexes(wd = depths-self.chndep, intervals = self.periods[stix],
                             dLimit = self.flood_depth_threshold)
    #-----------------------------------------------------------------------------------------------------------
    
    #-----------------------------------------------------------------------------------------------------------
    def update_vegetation(self, stix):
        return vegetation_model(HP = self.H[stix,:], MD = self.D[stix,:], rule = self.rule,
                                no_veg_cells = self.exclude)
#===============================================================================================================





#===============================================================================================================
class highFreqData(object):
    """ Object to manage high frequency datasets from hydrodynamic and sediment
    transpot simulations."""
    
    def __init__(self, Ndata=None, NoC=None, NoL=None, timestep=None, h=None,
                 v=None, Q=None, ssc=None):
        self.Ndata = Ndata
        self.NoC = NoC
        self.NoL = NoL
        self.timestep = timestep
        self.h = h
        self.v = v
        self.Q = Q
        self.ssc = ssc
    
    #-----------------------------------------------------------------------------------------------------------
    def load_h0(self):
        aux = initial_condition_read()
        self.h0 = aux['h0']
        
        if self.NoC is not None:
            if self.h0.size != self.NoC:
                print('\n ERROR! In: highFreqData.load_h0()')
                print(str(' > Number of records found in %s: %i,' % (fn_['init'], self.h0.size)) +
                    str(' is different from number of cells: %i\n' % self.NoC))
                exit()
    #-----------------------------------------------------------------------------------------------------------
    
    #-----------------------------------------------------------------------------------------------------------
    def load_native_outputs(self):
        
        #Number of records and timestep
        aux = initial_condition_read()
        self.finalTime = aux['t_final'][-1]    #time (s) of the last time-step in the simulation
        self.Dt_output = aux['Dt_output']      #Delta-time in the outputs of the hydrodynamic model
        self.NoTS = int(self.finalTime // self.Dt_output) #Number of Time-Steps
        
        #Number of cells and links
        aux = domain_read()
        self.NoC, self.NoL = aux['NoC'], aux['NoL']
        
        #Reading main setup options
        with open(fn_['setup'], 'r') as file:
            name, opt1, opt2 = [line[1:-2] for line in file.readlines()]
        
        if opt2 == 'N':
            #Loading water depths only
            self.load_native_series(variables=['h'])
        else:
            #Loading depths, velocities and discharges
            self.load_native_series()
    #-----------------------------------------------------------------------------------------------------------
    
    #-----------------------------------------------------------------------------------------------------------
    def load_native_series(self, variables=['h', 'v', 'Q']):
        name = get_hydrodynamic_id()
        
        for variable in variables:
            
            if variable == 'h':
                ext_prop = 'cells'
                ext_ncols = self.NoC
                filename = fn_['h'][0] + name + fn_['h'][1]
                
            elif variable == 'v':
                ext_prop = 'links'
                ext_ncols = self.NoL
                filename = fn_['v'][0] + name + fn_['v'][1]
                
            elif variable == 'Q':
                ext_prop = 'links'
                ext_ncols = self.NoL
                filename = fn_['Q'][0] + name + fn_['Q'][1]
                
            else:
                print('\n ERROR! In: highFreqData.load_native_series().')
                print(' > Cannot recognize variable "%s"\n' % variable)
                exit()
            
            aux = hydrodynamic_output_read(filename)
            
            if ((aux['Dt'] != self.Dt_output) or (aux['NoE'] != ext_ncols) or  (aux['Nts'] != self.NoTS)):
                print('\n ERROR loading %s: IN highFreqData.load_native_series()' % filename)
                print(' > Timestep in %s = %i; Timestep here = %i' % (fn_['init'], self.Dt_output, aux['Dt']))
                print(' > Number of %s in %s = %i; #%s here = %i' % (ext_prop, fn_['domain'], ext_ncols,
                      ext_prop, aux['NoE']))
                print(' > Expected number of records from %s = %i; #Ndata here = %i\n' % (fn_['init'],
                      self.NoTS, aux['Nts']))
                exit()
            
            if variable == 'h':
                self.h = aux['series']
            elif variable == 'v':
                self.v = aux['series']
            elif variable == 'Q':
                self.Q = aux['series']
    #-----------------------------------------------------------------------------------------------------------
    
    #-----------------------------------------------------------------------------------------------------------
    def save(self, filename, extension='pickle'):
        #Dump self object in a pickle file
        if extension == 'pickle' or extension == 'pkl':
            with open(filename+'.'+extension, 'wb') as file:
                pickle.dump(self, file)
#===============================================================================================================





#===============================================================================================================
class EGMproject(object):
    ''' Class used in the creation of a new domain and input files '''
    
    #-----------------------------------------------------------------------------------------------------------
    def build_hillslope_profiles(self, orientation='both'):
        ''' Sweep all rows and/or columns to find continuos sequence of cells connected by land-to-land
        link types, and store than to use in the hillslope diffusion model
        INPUTS:
        orientation [string] = 'both' to find longitudinal and latitudinal profiles, 'horizontal' to find only
            row-wise profiles and 'vertical' to find only column-wise profiles.
        OUTPUTS:
        Add/replace the self.profiles attributes with a list of arrays of sequence of cells in the same profile
        '''
        temp = {}
        sxf, syf, sfpf = self.x.flatten(), self.y.flatten(), self.floodplain.flatten()
        
        if orientation == 'both' or orientation == 'horizontal':
            #looking for sequence of linked cells in the same row
            for i in range(self.NoL):
                c1, c2 = self.cell1[i], self.cell2[i]
                #checking if cells are in the same row (same y-coordinate), if the linktype is land-to-land
                #and if both belong to the floodplain area
                if (syf[c1] == syf[c2]) and (self.lktype[i] == 61) and sfpf[c1] and sfpf[c2]:
                    #profile key name
                    key = 'y' + str(int(syf[c1]))
                    if key not in temp:
                        temp[key] = [[]]
                    #new profile in the current row
                    if c1 not in temp[key][-1]:
                        if len(temp[key][-1]) == 0:
                            temp[key][-1] = [c1,c2]
                        else:
                            temp[key].append([c1,c2])
                    else:
                        #add new cell to current profile
                        temp[key][-1].append(c2)
                    
        if orientation == 'both' or orientation == 'vertical':
            #looking for sequence of linked cells in the same column
            for i in range(self.NoL):
                c1, c2 = self.cell1[i], self.cell2[i]
                #checking if cells are in the same column (same x-coordinate), if the linktype is land-to-land
                #and if both belong to the floodplain area
                if (sxf[c1] == sxf[c2]) and (self.lktype[i] == 61) and sfpf[c1] and sfpf[c2]:
                    #profile key name
                    key = 'x' + str(int(sxf[c1]))
                    if key not in temp:
                        temp[key] = [[]]
                    #new profile in the current column
                    if c1 not in temp[key][-1]:
                        if len(temp[key][-1]) == 0:
                            temp[key][-1] = [c1,c2]
                        else:
                            temp[key].append([c1,c2])
                    else:
                        #add new cell to current profile
                        temp[key][-1].append(c2)
        
        self.profiles = [cells for p in temp.values() for cells in p]
    #-----------------------------------------------------------------------------------------------------------
    
    #-----------------------------------------------------------------------------------------------------------
    def create_links(self):
        ''' Create links between pairs of cells as the procedure in Simulaciones software. '''
        c1, c2 = [], []
        for row in range(0, self.nrows):
            for col in range(0, self.ncols):
                #Horizontal link
                if col != (self.ncols - 1):
                    c1.append(self.number[row,col])
                    c2.append(self.number[row,col+1])
                #Vertical link
                if row != (self.nrows - 1):
                    c1.append(self.number[row,col])
                    c2.append(self.number[row+1,col])
        c1 = np.array(c1, dtype=int)
        c2 = np.array(c2, dtype=int)
        return c1.size, c1, c2
    #-----------------------------------------------------------------------------------------------------------
    
    #-----------------------------------------------------------------------------------------------------------
    def finalize(self, folder=''):
        ''' Call a sequence of file-writing functions to dump the domain; special tiles; hillslope profiles;
        boundary conditions; and initial conditions data.
        INPUTS
        folder [str] = path to folder where the files will be created.
        '''
        #Adding a '/' at the end of the folder's string in case it is missing
        if len(folder) > 0:
            if folder[-1] != '/':
                folder += '/'
        
        #Creating the border status matrix
        border = np.zeros(self.NoC, dtype=int)
        active = np.array([b[0] for b in self.bound_at])
        border[active] = 1
        
        #Gauged link somewhere in the middle of the domain (this information doesn't affect the results)
        glink = [1, self.cell1[self.NoL//2], 1, self.cell2[self.NoL//2]]
        
        #Create file with link type and cells' data
        domain_write(NoC=self.NoC, NoL=self.NoL, gauge=glink, cell1=self.cell1, cell2=self.cell2,
            lktype=self.lktype, number=self.number.flatten(), border=border, z=self.z.flatten(),
            x=self.x.flatten(), y=self.y.flatten(), dx=self.dx.flatten(), dy=self.dy.flatten(),
            botwid=np.zeros(self.NoC), latslp=np.zeros(self.NoC), chndep=np.zeros(self.NoC),
            filename=folder+'vincul.dat')
        print('%-70s ... okay!' % (folder+'vincul.dat'))
        
        #Update parameters matrix
        define_parameters(params=self.params, cell1=self.cell1, cell2=self.cell2, lktype=self.lktype,
            Mann=self.Mann.flatten(), z=self.z.flatten(), x=self.x.flatten(), y=self.y.flatten(),
            dx=self.dx.flatten(), dy=self.dy.flatten(), botwid=np.zeros(self.NoC), latslp=np.zeros(self.NoC),
            chndep=np.zeros(self.NoC))
        
        #Create file with links' parameters
        parameters_write(NoL=self.NoL, cell1=self.cell1, cell2=self.cell2, lktype=self.lktype,
            params=self.params, cellchannel=(0.001, 0.001, 0.001), filename=folder+'param.dat')
        print('%-70s ... okay!' % (folder+'param.dat'))
        
        #Creating boundary condition water depths file
        boundary_depths_write(Dt_HT=self.bound_time_step, loc_HT=self.bound_at, HT=self.bound_h,
            filename=folder+'contar.dat')
        print('%-70s ... okay!' % (folder+'contar.dat'))
        
        #Creating run-time parameters / initial depths file
        initial_condition_write(ext_inp=(0,1), Dt_solver=self.solution_time_step,
            t_final=self.hydrodynamic_final_time, Dt_output=self.output_time_step, h0=self.initial_depth,
            filename=folder+'inicial.dat')
        print('%-70s ... okay!' % (folder+'inicial.dat'))
        
        #Special tiles file
        special_tiles_write(tiles=self.tiles, filename=folder+'special_tiles.dat')
        print('%-70s ... okay!' % (folder+'special_tiles.dat'))
        
        #Hillslope profiles file
        hillslope_profiles_write(profiles=self.profiles, filename=folder+'profiles.dat')
        print('%-70s ... okay!' % (folder+'profiles.dat'))        
    #-----------------------------------------------------------------------------------------------------------
    
    #-----------------------------------------------------------------------------------------------------------
    def initialize_from_dimensions(self, nrows, ncols):
        ''' Use the number of rows and number of columns to define initialize the domain properties. '''
        self.nrows = nrows
        self.ncols = ncols
        self.NoC = nrows * ncols
        self.x = np.zeros((nrows,ncols), dtype=float)
        self.y = np.zeros((nrows,ncols), dtype=float)
        self.z = np.zeros((nrows,ncols), dtype=float)
        self.dx = np.zeros((nrows,ncols), dtype=float)
        self.dy = np.zeros((nrows,ncols), dtype=float)
        self.Mann = np.zeros((nrows,ncols), dtype=float)
        self.number = np.arange(self.NoC).reshape((nrows,ncols))
        self.floodplain = np.ones((nrows,ncols), dtype=bool)
        self.NoL, self.cell1, self.cell2 = self.create_links()
        self.lktype = np.ones(self.NoL, dtype=int) * 61    #initialise as land-land link types
        self.params = np.zeros((self.NoL,12), dtype=float)
    #-----------------------------------------------------------------------------------------------------------
    
    #-----------------------------------------------------------------------------------------------------------
    def map_domain(self, fs=(10,4), msl=0.0):
        ''' Creates a pcolormesh plot of cell's elevation, and overlaps it with a scatter plot of cells'
        centre and links' positions
        '''
        grayscales = ['#404040', '#808080', '#bfbfbf', '#f2f2f2']
        fig, ax = pyplot.subplots(figsize=fs, dpi=100, constrained_layout=True)
        
        # make a colormap that has land and ocean clearly delineated and of the same length (256 + 256)
        colors_undersea = pyplot.cm.terrain(np.linspace(0, 0.17, 256))
        colors_land = pyplot.cm.terrain(np.linspace(0.25, 1, 256))
        all_colors = np.vstack((colors_undersea, colors_land))
        terrain_map = mcolors.LinearSegmentedColormap.from_list('terrain_map',all_colors)
        
        # creating a normalization, so that the elevations below 'msl' will look like water while above it
        #will look like land
        divnorm = mcolors.DivergingNorm(vmin=np.min(self.z)-0.01, vcenter=msl, vmax=np.max(self.z)+0.01)
        
        # create the coordinates matrices of cells' corners coordinates
        xFace = np.hstack( ([self.x[0,0]-0.5*self.dx[0,0]], self.x[0,:]+0.5*self.dx[0,:]) )
        yFace = np.hstack( ([self.y[0,0]+0.5*self.dy[0,0]], self.y[:,0]-0.5*self.dy[:,0]) )
        xp, yp = np.meshgrid(xFace, yFace)
        pcm = ax.pcolormesh(xp, yp, self.z, rasterized=True, norm=divnorm, cmap=terrain_map)
        
        # plotting cells centres in grayscales fading as we loop through each special tile area
        nofp = np.zeros((self.nrows, self.ncols), dtype=bool)    #an all-False matrix
        for tile in self.tiles:
            nofp = nofp|self.tiles[tile]["mask"]    # .or. operation
            ax.scatter(self.x[self.tiles[tile]["mask"]], self.y[self.tiles[tile]["mask"]], s=4,
                       c=grayscales.pop(0), marker='.', label=tile)
        ax.scatter(self.x[~nofp], self.y[~nofp], s=4, c='black', marker='.', label='Floodplain')
        
        # plotting links
        xc = self.x.flatten()
        yc = self.y.flatten()
        for i in range(self.NoL):
            xl = 0.5 * (xc[self.cell1[i]] + xc[self.cell2[i]])
            yl = 0.5 * (yc[self.cell1[i]] + yc[self.cell2[i]])
            if self.lktype[i] == 11:
                ax.plot([xl],[yl], c='blue', marker='+', markersize=9)
            else:
                if xc[self.cell1[i]] == xc[self.cell2[i]]:
                    ax.plot([xl],[yl], c='yellow', marker='_')
                else:
                    ax.plot([xl],[yl], c='yellow', marker='|')
        
        # final settings
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        fig.colorbar(pcm, shrink=0.6, extend='both', label='Elevation [m]')
        
        return fig, ax
    #-----------------------------------------------------------------------------------------------------------
    
    #-----------------------------------------------------------------------------------------------------------
    def place_embankment_at_x(self, atx, rows):
        """ Find the pairs of horizontally linked cells where the embankment-coordinate lay between their
        center's position
        INPUTS:
            atx |float| = embankment's x coordinate
            rows |list of integers| = rows of the 2D domain where the embankment will cross through
        RETURN:
            pairs |list of linked cells| = list with the number of linked cells that will have their link
                removed to represent the embankment
        OUTPUT:
            This function will remove the links listed in 'pairs' from all self attributes related to link
            properties.
        """
        # Using the first row as reference, find the columns immediately before and after the embankment
        before = np.max(np.where(self.x[rows[0],:] < atx)[0])
        after = np.min(np.where(self.x[rows[0],:] > atx)[0])
        #Listing cells pairs of cells at columns (before,after) in the given list of rows
        pairs = []
        for row in rows:
            pairs.append( [self.number[row,before], self.number[row,after]] )
        #Removing links where there is an embankment
        self.remove_links(pairs)
        #Return pairs of cells separated by the embankment
        return pairs
    #-----------------------------------------------------------------------------------------------------------
    
    #-----------------------------------------------------------------------------------------------------------
    def place_embankment_at_y(self, aty, cols):
        """ Find the pairs of vertically linked cells where the embankment-coordinate lay between their center's
        position
        INPUTS:
            aty |float| = embankment's y coordinate
            cols |list of integers| = columns of the 2D domain where the embankment will cross through
        RETURN:
            pairs |list of linked cells| = list with the number of linked cells that will have their link
                removed to represent the embankment
        OUTPUT:
            This function will remove the links listed in 'pairs' from all self attributes related to link
            properties.
        """
        # Using the first column as reference, find the rows immediately below and above the embankment
        #Remember that row-order and y-axis are inversely orientated
        below = np.min(np.where(self.y[:,cols[0]] < aty)[0])
        above = np.max(np.where(self.y[:,cols[0]] > aty)[0])
        #Listing cells pairs of cells at rows (above,below) in the given list of columns
        pairs = []
        for col in cols:
            pairs.append( [self.number[col,above], self.number[col,below]] )
        #Removing links where there is an embankment
        self.remove_links(pairs)
        #Return pairs of cells separated by the embankment
        return pairs
    #-----------------------------------------------------------------------------------------------------------
    
    #-----------------------------------------------------------------------------------------------------------
    def remove_links(self, pairs):
        ''' Find those links establish for each pair of values given in 'pairs' and remove them from all
        self attributes related to link's information. '''
        bool_links = np.ones(self.NoL, dtype=bool)
        for p in pairs:
            i = np.where((self.cell1 == p[0]) & (self.cell2 == p[1]))
            bool_links[i] = False
        self.cell1 = self.cell1[bool_links]
        self.cell2 = self.cell2[bool_links]
        self.lktype = self.lktype[bool_links]
        self.params = self.params[bool_links,:]
        aux = self.NoL
        self.NoL = len(self.cell1)
        print('\n%i links removed from domain\n' % (aux - self.NoL))
    #-----------------------------------------------------------------------------------------------------------
    
    #-----------------------------------------------------------------------------------------------------------
    def set_boundary_depths(self, wl):
        """ Creates a matrix of water depths at each boundary-link
        INPUTS:
            wl [array] = time series of water levels to be applied at all boundary-links
        OUTPUT:
            It create/replace the self attribute 'bound_h', which is the water depths matrix with shape (n,m)
        where n is the number of records and m the number of boundary-links
        """
        zflat = self.z.flatten()
        wd = np.transpose(np.tile(wl, (len(self.bound_at),1)))
        wd = np.subtract(wd, [zflat[link[0]] for link in self.bound_at])
        wd[wd < 0] = 0.0
        self.bound_h = wd
    #-----------------------------------------------------------------------------------------------------------
    
    #-----------------------------------------------------------------------------------------------------------
    def set_coordinates(self, xWest=0.0, ySouth=0.0):
        """ Set the self.x and self.y coordinates based on the length of each column and row.
        Pay attention that x increase from left to righ, while y increase from bottom to top. Therefore,
        the rows need to be flipped to match this orientation.
        INPUTS:
            xWest [float] = define the x-coordinate of the West face of the cells in the first column
            ySouth [float] = define the y-coordinate of the South face of the cells in the bottom row
        """        
        for i in range(self.ncols):
            xEast = xWest + self.dx[0,i]
            self.x[:,i] = 0.5 * (xEast + xWest)
            xWest = xEast
            
        for j in range(self.nrows-1,-1,-1):
            yNorth = ySouth + self.dy[j,0]
            self.y[j,:] = 0.5 * (ySouth + yNorth)
            ySouth = yNorth
    #-----------------------------------------------------------------------------------------------------------
    
    #-----------------------------------------------------------------------------------------------------------
    def set_culvert(self, cell1, cell2, base_width, gate_opening, base_elev=None, base_coef=0.8,
        middle_width=None, middle_elev=None, middle_coef=0.8, top_width=None, top_elev=None, top_coef=0.8,
        downstream_elev=None, upstream_elev=None):
        """ Replace the link between cell1 and cell2 to a culvert with gate one.
        INPUTS:
            cell1 [int] = cell's number of the first cell in the link
            cell2 [int] = cell's number of the second cell in the link
            base_width [float] = culvert's 1st step width (metres)
            gate_opening [float] = heigh between the 1st step bottom elevation and the gate's level (metres)
            base_elev [float] = bottom elevation of culvert's 1st step (metres). If is None, use the average
                level between the linked cells
            base_coef [float] = discharge coefficient of culvert's 1st step (adim.)
            middle_width, middle_elev, middle_coef [floats] = equivalent parameters for culvert's 2nd step
            top_width, top_elev, top_coef [floats] = equivalent parameters for culvert's 3rd step
            downstream_elev, upstream_elev [floats] = sill elevation in the first and second link cell
                respectively
        RETURN:
            link [int] = link's number where the culvert was set
        OUTPUTS:
            Replace the parameters in self.params[links,:] and change the link-type to 10
        """
        link, = np.where((self.cell1 == cell1) & (self.cell2 == cell2))
        if middle_width is None:
            middle_width = base_width + 0.02
        if top_width is None:
            top_width = middle_width + 0.02
        if base_elev is None:
            base_elev = 0.5 * (self.z.flatten()[cell1] + self.z.flatten()[cell2])
        if middle_elev is None:
            middle_elev = base_elev + 10.
        if top_elev is None:
            top_elev = middle_elev + 10.
        if downstream_elev is None:
            downstream_elev = base_elev
        if upstream_elev is None:
            upstream_elev = base_elev
        self.params[link,:] = base_elev, base_width, base_coef, middle_elev, middle_width, middle_coef, \
            top_elev, top_width, top_coef, downstream_elev, upstream_elev, gate_opening
        self.lktype[link] = 11
        return link
    #-----------------------------------------------------------------------------------------------------------
    
    #-----------------------------------------------------------------------------------------------------------
    def set_initial_depths(self, iwl):
        ''' Creates an array of water depth over the domain given the input-water-level (iwl) at t=0 '''
        wd = np.ones(self.NoC, dtype=float) * iwl - self.z.flatten()
        wd[wd < 0] = 0.0
        self.initial_depth = wd
    #-----------------------------------------------------------------------------------------------------------
    
    #-----------------------------------------------------------------------------------------------------------
    def set_special_tiles(self, name, indexes, mann=0.035):
        ''' Create/add a new special area of the domain in the self.tiles dictionary
        INPUTS:
            name [string] = key-string to access the tiles settings
            indexes [list or array] = sequence of cells' indexes within the new area
        OUTPUTS:
        This function will create/update the attribute self.tiles, which is a Python dictionary.
        .tiles[name] stores another dict:
            .tiles[name]['cells'] |array| = array with cells' number/index in the given area
            .tiles[name]['roughness'] |float| = value of Manning's roughness coefficient in the given area
            .tiles[name]['mask'] |2D-array| = boolean array with shape (self.nrows, self.ncols) to access the
                cells in the given area throughout the 2D-domain setup.
        '''
        if isinstance(indexes, list):
            flat_list = [item for sublist in indexes for item in sublist]
            indexes = np.array(flat_list, dtype=int)
        elif isinstance(indexes, np.ndarray):
            indexes = indexes.flatten()
        
        if not hasattr(self, 'tiles'):
            self.tiles = {}
        
        self.tiles[name] = {'cells': indexes}   #if self.tiles already had the <name> key, it is replaced.
        self.tiles[name]['roughness'] = mann
        self.tiles[name]['mask'] = np.isin(self.number, indexes)
        self.Mann[self.tiles[name]['mask']] = mann
        self.floodplain[self.tiles[name]['mask']] = False
#===============================================================================================================





























#===============================================================================================================
# FILE MANIPULATION FUNCTIONS
#===============================================================================================================
def boundary_depths_read(filename=None, return_as_matrix=False, properties=None):
    """ Reads the hydrodynamic model's file with HxT boundary conditions (i.e. water depths time series)
    INPUTS:
        filename |string| = HxT boundary condition file's name
        return_as_matrix |bool| = if set to True, data['HT'] will be a 2D np.ndarray, otherwise a pd.DataFrame
        properties |str or list| = key or list of required keys (see data-dictionary below). Returns full data
            if it is None.
    RETURN:
        data |dict|
            data['Nr_HT'] = number of records in the HxT series
            data['Dt_HT'] = delta time (s) between records in the HxT series
            data['nun_HT'] = number of boundary cells/links where HxT are applied
            data['loc_HT'] = np.ndarray((nun_HT,3), int) to store:
                [:,0] index of the cell where the depths will be given by the HxT series
                [:,1] index of the linked cell where the HxT is applied
                [:,2] the status of the boundary link, being 1 for active and 0 for inactive.
            data['HT'] = pd.DataFrame of water depths series at each control cell (set as DF columns), indexed
                by time stamp (in seconds)
    """
    # Initialise data dictionary, read the file content and take the information in the first line
    data = {'Nr_HT': None, 'Dt_HT': None, 'nun_HT':None}
    if filename is None:
        name = fn_['bounds']
    else:
        name = filename
    with open(name, 'r') as file:
        aux = file.readlines()
        data['Nr_HT'], data['Dt_HT'], data['nun_HT'] = list(map(int, aux[0].split()))
    
    # Create matrices of boundary location and depths series
    data['loc_HT'] = np.zeros((data['nun_HT'], 3), dtype=int)
    tmp = np.zeros((data['Nr_HT'], data['nun_HT']), dtype=float) * np.nan
    l = 1    #line's index (of the opened file)
    
    # Looping throughout the boundary locations
    for i in range(data['nun_HT']):
        data['loc_HT'][i,:] = list(map(int, aux[l].split()))
        data['loc_HT'][i,0:2] -= 1    #subtract 1 from cell's number (FORTRAN) to turn into cell's index (PYTHON)
        
        for j in range(data['Nr_HT']):
            l += 1
            tmp[j,i] = float(aux[l])
        
        l += 1
    
    if return_as_matrix:
        #Called to return data as 2D np.array
        data['HT'] = tmp
        
    else:
        # Creating list with time stamps, then the DataFrame with all water depths series
        stamps = np.arange(data['Nr_HT']) * data['Dt_HT']
        data['HT'] = pd.DataFrame(tmp, columns=data['loc_HT'][:,0], index=stamps)
    
    # Finish
    if isinstance(properties, str):
        return data[properties]
    elif isinstance(properties, list):
        return [data[p] for p in properties]
    elif properties is None:
        return data
    else:
        raise ValueError('\n Could not understand the returning properties type\n')
#---------------------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------------
def boundary_depths_update(Dt_HT=None, loc_HT=None, HT=None, levels=None, elevs=None, filename=None):
    """ Update some components of the hydrodynamic model's file with HxT boundary conditions (i.e. water depths
    time series). It reads the file and replace/recompute those components provided in the arguments (that are
    not None) and overwrite the file.
    INPUTS:
        Dt_HT |int| = delta time between records in the HxT series (seconds)
        loc_HT |array((nun_HT,3), int)| = array with information on each boundary point: control cell,
            linked cell, status
        HT = it can be an np.ndarray((Nr_HT, nun_HT), float), or a pd.DataFrame with same dimensions, where
            Nr_HT is the number of records in the HxT series and nun_HT is the number of HxT boundary points.
        levels |array((Nr_HT), float)| = water level series to be applied at all HxT boundary points. When
            provided, elevs must be provided as well.
        elevs |array((NoC), float)| = array with bottom elevation of all NoC cells in the domain
        filename |str| = if provided will replace the value in fn_['bounds'] before writing the updated file.
    """
    # Read file returning the HxT series in a 2D np.ndarray instead of a pd.DataFrame
    data = boundary_depths_read(return_as_matrix=True)
    
    # If Dt_HT was provided, replace 'Dt_HT' in read data
    if Dt_HT is not None:
        data['Dt_HT'] = Dt_HT
        
    # If loc_HT was provided, replace 'loc_HT' in the read data and review 'num_HT' as well
    if loc_HT is not None:
        data['loc_HT'] = loc_HT
        data['nun_HT'] = loc_HT.shape[0]
    
    # If HT was provided replace 'HT' matrix in the read data
    if HT is not None:
        data['HT'] = HT
    
    # If levels and elevs were provided, compute water depths series and replace 'HT' in read data
    if (levels is not None) and (elevs is not None):
        wd = np.transpose(np.tile(np.array(levels), (data['nun_HT'],1)))
        wd = np.subtract(wd, [elevs[link[0]] for link in data['loc_HT']])
        wd[wd < 0] = 0.0
        data['HT'] = wd
        
    # If filename was provided, replace it in _fn dictionary
    if filename is not None:
        fn_['bounds'] = filename
    
    # Write file
    boundary_depths_write(data['Dt_HT'], data['loc_HT'], data['HT'])
    
    # Finish
    return True
#---------------------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------------
def boundary_depths_write(Dt_HT, loc_HT, HT, filename=None):
    """ Write the hydrodynamic model's file with HxT boundary conditions (i.e. water depths time series)
    INPUTS:
        Dt_HT |int| = delta time between records in the HxT series
        loc_HT |array((nun_HT,3), int)| = matrix with location information (control cell, linked cell, status)
            on each HxT boundary point
        HT = it can be an np.ndarray((Nr_HT, nun_HT), float), or a pd.DataFrame with same dimensions, where
            Nr_HT is the number of records in the HxT series and nun_HT is the number of HxT boundary points.
        filename |string| = HxT boundary condition file's name
    """
    # Open file to write
    if filename is None:
        f = open(fn_['bounds'], 'w')
    else:
        f = open(filename, 'w')
    
    # Dealing with data in an numpy ndarray
    if type(HT).__name__ == 'ndarray':
        # First line
        f.write('%i %i %i\n' % (HT.shape[0], Dt_HT, loc_HT.shape[0]))
        # Looping through the boundary points
        for i in range(loc_HT.shape[0]):
            #control cell, linked cell, boundary status
            f.write('%i %i %i\n' % (loc_HT[i][0]+1, loc_HT[i][1]+1, loc_HT[i][2]))
            #writing HxT data of current boundary point
            for j in range(HT.shape[0]):
                f.write('%10.6f\n' % HT[j,i])
    
    # Dealing with data in an pandas DataFrame
    if type(HT).__name__ == 'DataFrame':
        # First line
        f.write('%i %i %i\n' % (HT.index.size, Dt_HT, loc_HT.shape[0]))
        # Looping through the boundary points
        for i in range(loc_HT.shape[0]):
            #control cell, linked cell, boundary status
            f.write('%i %i %i\n' % (loc_HT[i][0]+1, loc_HT[i][1]+1, loc_HT[i][2]))
            #writing HxT data of current boundary point
            for j in range(HT.index.size):
                f.write('%10.6f\n' % HT.iloc[j,i])
    
    # End of file, close it.
    f.close()
    
    # Finish
    return True
#---------------------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------------
def domain_read(filename=None):
    """ Reads the hydrodynamic model's file with domain properties and link type
    INPUTS:
        filename |string| = name of domain's properties file
    RETURN: data |dict|
        data['NoC'], data['NoL'] |int| = number of cells and links in the domain, respectively
        data['gauge'] = array of integers to set 'gauged' links
        data['cell1'], _['cell2'], _['lktype'] |np.array((NoL), int)| = outlet cell's index, inlet cell's index
            and link-type.
        data['number'] |np.array((NoC), int)| = cell's index.
        data['border'] |np.array((NoC), int)| = 0/1 (False/True) values to indicate if the cell is in the
            boundary condition
        data['x'], _['y'], _['z'], _['dx'], _['dy'] |np.array((NoC), float)| = x-coordinate, y-coordinate,
            bottom elevation, x-length, y-length.
        data['botwid'], _['latslp'], _['chndep'] |np.array((NoC), float)| = cell's channel properties:
            botwid = bottom width, latslp = lateral slope, chndep = channel depth.            
    """
    # Initialise data dictionary and read the file content
    data = {}
    
    if filename is None:
        name = fn_['domain']
    else:
        name = filename
        
    with open(name, 'r') as file:
        aux = [line.split() for line in file]
    
    # Number of cells and number of links
    data['NoC'], data['NoL'] = map(int, aux[0])
    
    # Gauged links
    data['gauge'] = list(map(int, aux[1]))
    
    #Linked cells and link type
    data['cell1'] = np.zeros(data['NoL'], dtype=int)
    data['cell2'] = np.zeros(data['NoL'], dtype=int)
    data['lktype'] = np.zeros(data['NoL'], dtype=int)
    for i in range(data['NoL']):
        data['cell1'][i], data['cell2'][i], data['lktype'][i] = map(int, aux[i+2])
        
    #Cells data
    data['number'] = np.zeros(data['NoC'], dtype=int)
    data['border'] = np.zeros(data['NoC'], dtype=int)
    data['x'] = np.zeros(data['NoC'], dtype=float)
    data['y'] = np.zeros(data['NoC'], dtype=float)
    data['z'] = np.zeros(data['NoC'], dtype=float)
    data['dx'] = np.zeros(data['NoC'], dtype=float)
    data['dy'] = np.zeros(data['NoC'], dtype=float)
    data['botwid'] = np.zeros(data['NoC'], dtype=float)
    data['latslp'] = np.zeros(data['NoC'], dtype=float)
    data['chndep'] = np.zeros(data['NoC'], dtype=float)
    for i in range(data['NoC']):
        j = 2 + data['NoL'] + i
        data['number'][i], data['border'][i] = int(aux[j][0]), int(aux[j][1])
        data['z'][i], data['x'][i], data['y'][i], data['dx'][i], data['dy'][i], data['botwid'][i], \
        data['latslp'][i], data['chndep'][i] = map(float, aux[j][2:10])
    
    # Change from FORTRAN indexation (first position is at [1]) to Python index (first = [0])
    data['cell1'] -= 1
    data['cell2'] -= 1
    data['number'] -= 1
    
    # Finish
    return data
#---------------------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------------
def domain_update(**kwargs): #(NoC, NoL, gauge, cell1, cell2, lktype, number, border, x, y, z, .. filename=None):
    """ Update the hydrodynamic model's file with domain properties and link type. It reads the file, replace
    the components provided (key:value in kwargs that match with the expected inputs) and overwrite the file.
    INPUTS:
        Same as domain_write() excepting 'NoL', 'cell1', 'cell2', 'lktype' and 'filename'. The first four can
            be updated via parameters_update(). The last works as explained below
        filename |str| = if provided will replace the value in fn_['domain'] before writing the updated file.
    
    * kwargs is a dictionary with all "name=value" entries passed to this function
    """
    # Read file
    data = domain_read()
    
    # Replacing provided components
    for key in ['NoC', 'gauge', 'number', 'border', 'x', 'y', 'z', 'dx', 'dy', 'botwid', 'latslp', 'chndep']:
        if key in kwargs:
            data[key] = kwargs[key]
    
    # If filename was provided, replace it in fn_ dictionary
    if 'filename' in kwargs:
        fn_['domain'] = kwargs['filename']
        del(kwargs['filename'])
    
    # Write file
    domain_write(**data)
    
    # Finish
    return True
#---------------------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------------
def domain_write(**kwargs): #(NoC, NoL, gauge, cell1, cell2, lktype, number, border, x, y, z, .. filename=None):
    """ Write the hydrodynamic model's file with domain properties and link type
    INPUTS:
        NoC, NoL |int| = number of cells and links in the domain, respectively
        gauge |array-like| = a list/tuple/ndarray of integers to set links where the results will be printed on
            screen (not really usefull)
        cell1, cell2, lktype |array-like| = sequence of NoL integers of outlet cell's index, inlet cell's index
            and link-type. This function will add 1 to cell1 and cell2 to change to FORTRAN indexation
        number |array-like| = sequence of NoC integers with cell's index. This function adds 1 to number to
            change to FORTRAN indexation
        border |array-like| = sequence of NoC 0/1 (False/True) values to indicate if the cell is in the boundary
            condtion
        x, y, z, dx, dy |array-like| = sequence of NoC values with the following cell's properties:
            x-coordinate, y-coordinate, z=bottom elevation, x-length, y-length.
        botwid, latslp, chndep |array-like| = sequence of NoC values with cell's channel properties:
            botwid = bottom width, latslp = lateral slope, chndep = channel depth. All these properties should
            be 0 (zero) if the cell is land-type
        filename |string| = path to file with domain properties
    
    * kwargs is a dictionary with all "name=value" entries passed to this function
    """
    # Open file to write
    if 'filename' in kwargs:
        f = open(kwargs['filename'], 'w')
    else:
        f = open(fn_['domain'], 'w')
    
    # Write number of cells and number of links
    f.write('%i   %i\n' % (kwargs['NoC'], kwargs['NoL']))
    
    # Set the gauged link(s)
    for i in range(len(kwargs['gauge'])):
        f.write(' %i' % kwargs['gauge'][i])
    f.write('\n')
    
    # For each link in the domain...
    for i in range(kwargs['NoL']):
        #... write the link's cells numbers and its link type code
        #adds 1 to cell's number to change cell counting from 1 on (instead of 0)
        f.write('%5i %5i %i\n' % (kwargs['cell1'][i]+1, kwargs['cell2'][i]+1, kwargs['lktype'][i]))
        
    # For each cell ...
    for i in range(kwargs['NoC']):
        #... write the cell number, the border status,
        f.write('%5i %1i' % (kwargs['number'][i]+1, kwargs['border'][i]))
        #cell's bottom elevation and its center (x,y) coordinates
        f.write(' %10.6f %10.3f %10.3f' % (round(kwargs['z'][i],6), kwargs['x'][i], kwargs['y'][i]))
        #cell's x-length and y-length
        f.write(' %7.3f %7.3f' % (kwargs['dx'][i], kwargs['dy'][i]))
        #cell's channel properties: bottom width, lateral slope and channel depth
        f.write(' %8.4f %8.4f %8.4f\n' % (kwargs['botwid'][i], kwargs['latslp'][i], round(kwargs['chndep'][i],4)))
        
    # End of file, close it
    f.close()
    
    #Finish
    return True
#---------------------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------------
def egm_settings_read(filename):
    """ Read (or load) file with options/settings to run an EGM simulation.
    INPUTS:
        filename |str| = file's path/name. It can be a .csv, .json or .pickle file
    RETURN:
        options |dict| = dictionary with the EGM simulation settings
    """
    options = {}
    
    if filename.endswith('csv'):
        # Reading options from CSV file
        file = open(filename, 'r')
        
        for line in file:            
            line = line.rstrip()
            line = line.split(',')
            
            if len(line[-1]) == 0:
                line = line[:-1]
            
            if len(line) == 2:
                #single pair of key,value
                options[line[0]] = line[1]
                
            elif len(line) == 3:
                #parent key, child key, value
                if line[0] not in options:
                    options[line[0]] = {line[1]:line[2]}
                else:
                    options[line[0]][line[1]] = line[2]
        
        file.close()
        
    elif filename.endswith('json'):
        # Loading options from JSON file
        file = open(filename, 'r')
        options = json.load(file)
        file.close()
        
    elif filename.endswith('pickle') or filename.endswith('pkl'):
        # Loading options from PICKLE file
        file = open(filename, 'rb')
        options = pickle.load(file)
        file.close()
        
    else:
        print(str("\n ERROR: In: egm_settings_read()\nThe input file: %s doesn't" % filename) +
                  " have a 'csv', 'json' or 'pickle' extension.\n")
        exit()
    
    #End
    return options
#---------------------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------------
def hillslope_profiles_read(filename=None):
    """ Read the profiles to be used in the Hillslope Diffusion model.
    INPUTS:
        filename = string with profile file's path/name. If it is none, read from fn_['profiles']
    RETURN:
        profiles |list| = list with arrays containing the cells' index that compose each single hillslope profile
        delta_x |array| = distance between cells on each profile
    """
    # Read file content
    if filename is None:
        name = fn_['profiles']
    else:
        name = filename
    with open(name, 'r') as f:
        fc = f.readlines()
    
    # Initialisations
    i, profiles, delta_x = 0, [], []
    
    # Looping through the file until reach its end
    while True:
        line = fc[i].split()
        profID, dist, N = line[0], float(line[1]), int(line[2])
        delta_x.append(dist)
        cells = list(map(int, fc[i+1:i+N+1]))
        profiles.append(np.array(cells, dtype=int))
        i = i + N + 1
        if i >= len(fc):
            break
    
    return profiles, np.array(delta_x, dtype=float)  
#---------------------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------------
def hillslope_profiles_write(profiles, delta_x=10., filename=None):
    """ Write the profiles (sequence of cells' index in the same line) used in the Hillslope Diffusion model.
    INPUTS:
        filename |string| = output file name. If it is None use fn_['profiles']
        profiles |list| = list of array-like sets containing the cells' index that compose each single
            hillslope profile
        delta_x |value or array-like| = distance between cells in the profile. If it is passed as a single value,
            an array repeating its value for the same number of profiles will be created
    """
    # Turning delta_x in a list with the same value for each profile
    if isinstance(delta_x, float) or isinstance(delta_x, int):
        delta_x = np.ones(len(profiles), dtype=float) * delta_x
        
    # Open file to write
    if filename is None:
        f = open(fn_['profiles'], 'w')
    else:
        f = open(filename, 'w')
        
    # For each profile ...
    for i in range(len(profiles)):
        #... write an ID (will not be used anywhere, it just to make it easier
        #to identify each profile), the delta_x and the number of cells in the
        #current profile
        ncells = len(profiles[i])
        f.write('Profile%3.3i  %.1f  %i\n' % (i+1, delta_x[i], ncells))
        # write each cell number in the profile
        for j in range(ncells):
            f.write('%i\n' % profiles[i][j])
            
    # End of file, close it
    f.close()
    
    # Finish
    return True
#---------------------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------------    
def hydrodynamic_output_read(filename):
    """ Load data from a hydrodynamic model's output file (depth, velocities or flow)
    INPUT:
        filename |str| = path/name of the output files. Its file format is quite simple: At the first line it
            must have had the number of cells or links (ncols), followed by the delta time between records.
            From the second line on, it should have all the data like in a flattened array with shape
            (nrows, ncols), although nrows is not known before-hand
    RETURN:
        data |dict| = dictionary with the following items
            data['Dt']  |float| = delta time between records
            data['NoE'] |int|   = number of elements (= NoC for depths; = NoL for velocity or discharge)
            data['Nts'] |int|   = number of time steps. Computed based on tje amount of data read from file
            data['series'] |np.array((Nts,NoE), float)| the output data as np.array((Nts,NoE)).flatten()
    """
    fct = np.fromfile(filename, sep=' ')
    data = {'NoE': int(fct[0]), 'Dt': fct[1]}
    data['Nts'] = (fct.size - 2) // data['NoE']
    data['series'] = fct[2:].reshape((data['Nts'],data['NoE']))
    
    return data
#---------------------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------------
def initial_condition_read(filename=None, properties=None):
    """ Reads the hydrodynamic model's file with runtime settings and water depths at t = 0
    INPUTS:
        filename |string| = name of runtime settings and initial condition file
        properties |str or list| = key or list of required keys (see data-dictionary below). Returns full data
            if it is None.
    RETURN:
        data |dict|
            data['ext_inp'] = list with indicator values (0/False or 1/True) for external inputs (discharge
                and rainfall)
            data['Dt_solver'] = hydrodynamic model's solution time-steps (s). One for each one of the four
                sub-intervals
            data['t_final'] = list with time (s) at the end of each one of the four sub-intervals in the
                hydrodynamic simulation
            data['Dt_output'] = delta time between records in the hydrodynamic model's output files
            data['h0'] = array with initial water depths on each cell of the domain
    """
    # Initialise data dictionary and read the file content
    data = {}
    if filename is None:
        name = fn_['init']
    else:
        name = filename
    with open(name, 'r') as file:
        aux = file.readlines()
    
    # External inputs switch
    data['ext_inp'] = [int(x) for x in aux[0].split()]
    
    # Info in the second line
    line = [float(x) for x in aux[1].split()]    
    data['Dt_solver'] = line[0:4]    #solution time-steps
    data['t_final']   = line[4:8]    #final time of each sub-interval
    data['Dt_output'] = line[8]      #time delta between records in the output files
    
    # Initial condition
    data['h0'] = np.array([item for line in aux[2:] for item in line.split()], dtype=float)
    
    # Finish
    if isinstance(properties, str):
        return data[properties]
    elif isinstance(properties, list):
        return [data[p] for p in properties]
    elif properties is None:
        return data
    else:
        raise ValueError('\n Could not understand the returning properties type\n')
#---------------------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------------
def initial_condition_update(ext_inp=None, Dt_solver=None, t_final=None, Dt_output=None, h0=None, filename=None):
    """ Update some components of the hydrodynamic model's file with runtime settings and initial water depths.
    It reads the file and replace those components provided in the arguments (that are not None) and overwrite
    the file.
    
    *array-like = list, tuple or np.ndarray
    
    INPUTS:
        ext_inp |array-like*| = pair of integers to indicate the use of external inputs. The integers must be
            0 (False) or 1 (True). The first element is related to input discharge and the second to rainfall.
        Dt_solver |array-like| = hydrodynamic model's solution time-steps (s). One for each one of the four
            sub-intervals
        t_final |array-like| = time (s) at the end of each one of the four sub-intervals in the hydrodynamic
            simulation
        Dt_output |int or float| = delta time between records in the hydrodynamic model's output files
        h0 |array-like| = initial water depths on each cell of the domain
        filename |str| = if provided will replace the value in fn_['init'] before writing the updated file.
    """
    # Read file
    data = initial_condition_read()
    
    # If ext_inp was provided, replace 'ext_inp' in read data
    if ext_inp is not None:
        data['ext_inp'] = ext_inp
        
    # If Dt_solver was provided, replace 'Dt_solver' in the read data
    if Dt_solver is not None:
        data['Dt_solver'] = Dt_solver
    
    # If t_final was provided replace 't_final' the read data
    if t_final is not None:
        data['t_final'] = t_final
    
    # If Dt_output was provided replace 'Dt_output' the read data
    if Dt_output is not None:
        data['Dt_output'] = Dt_output
    
    # If h0 was provided, replace 'h0' in read data
    if h0 is not None:
        data['h0'] = h0
        
    # If filename was provided, replace it in fn_ dictionary
    if filename is not None:
        fn_['init'] = filename
    
    # Write file
    initial_condition_write(data['ext_inp'], data['Dt_solver'], data['t_final'], data['Dt_output'], data['h0'])
    
    # Finish
    return True
#---------------------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------------
def initial_condition_write(ext_inp, Dt_solver, t_final, Dt_output, h0, filename=None):
    """ Write the hydrodynamic model's file with runtime settings and water depths at t = 0.
    Remember that the model divides the simulation period in 4 sub-intervals, each one with its own solution
    time-step and final time.
    
    *array-like = list, tuple or np.ndarray
    
    INPUTS:
        ext_inp |array-like*| = pair of integers to indicate the use of external inputs. The integers must be
            0 (False) or 1 (True). The first element is related to input discharge and the second to rainfall.
        Dt_solver |array-like| = hydrodynamic model's solution time-steps (s). One for each one of the four
            sub-intervals
        t_final |array-like| = time (s) at the end of each one of the four sub-intervals in the hydrodynamic
            simulation
        Dt_output |int or float| = delta time between records in the hydrodynamic model's output files
        h0 |array-like| = initial water depths on each cell of the domain
        filename |string| = path to file with runtime settings and initial condition
    """
    # Open file to write
    if filename is None:
        f = open(fn_['init'], 'w')
    else:
        f = open(filename, 'w')
    
    # 0/1 = False/True for external discharge and rainfall
    f.write('%i %i\n' % (ext_inp[0], ext_inp[1]))
    
    # Solution time-step in each sub-interval
    line = str('%.3f %.3f %.3f %.3f' % tuple(Dt_solver))
    
    # Final time in each sub-interval
    line += str('   %.3f %.3f %.3f %.3f' % tuple(t_final))
    
    # Delta time between records in the output files
    line += str('   %.3f\n' % Dt_output)
    f.write(line)
    
    #Writing water depths, 10 records per line
    for i in range(len(h0)):
        if i > 0 and i%10 == 0:
            f.write('\n')
        f.write(' %.6f' % h0[i])
    
    # End of file, close it.
    f.close()
    
    # Finish
    return True
#---------------------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------------
def parameters_read(param_filename=None, domain_filename=None):
    """ Reads the hydrodynamic model's file with links' parameters. It also reads the domain file to get some
    parameters-related information
    INPUTS:
        param_filename |string| = name of links' paramete file. If is None, use fn_['param']
        domain_filename |string| = name of links' type file. If is None, use fn_['domain']
    RETURN: data |dict|
     > From domain properties file
        data['NoL'] |int| = number of links in the domain
        data['lktype'] |np.array((NoL), int)| = link type code
     > From parameters file
        data['cell1'], data['cell2'] |np.array((NoL), int)| = outlet and inlet cell's index.
        data['params'] |np.array((NoL,12), float)| = links' parameter matrix
        data['cellchannel'] |np.array((3), float)| = land cells inner channel parameters: bottom width,
            lateral slope and channel depth
    """
    # Reading domain's properties
    dom = domain_read(domain_filename)
        
    # Initialise data dictionary with link's data from domain file
    data = {'NoL': dom['NoL'], 'lktype': dom['lktype'], 'cell1': np.zeros(dom['NoL'], dtype=int),
            'cell2': np.zeros(dom['NoL'], dtype=int), 'params': np.zeros((dom['NoL'],12), dtype=float),
            'cellchannel': np.zeros(3, dtype=float)}
    
    # Opening parameter's file
    if param_filename is None:
        name = fn_['param']
    else:
        name = param_filename
        
    with open(name, 'r') as file:
        aux = [line.split() for line in file]
    
    #Parameters data    
    for i in range(data['NoL']):
        data['cell1'][i] = int(aux[i][0]) - 1    #subtracts one to change from Fortran to Python indexation
        data['cell2'][i] = int(aux[i][1]) - 1
        
        if data['cell1'][i] != dom['cell1'][i] or data['cell2'][i] != dom['cell2'][i]:            
            print('\n ERROR: Cells number are different for the same link.')
            print('PARAM.DAT, LINE', i+1, ':', aux[i])
            print('VINCUL.DAT, LINE', i+3,':', [dom['cell1'][i], dom['cell2'][i], dom['lktype'][i]], '\n')
            raise ValueError()
        
        for j in range(len(aux[i])-2):
            data['params'][i,j] = float(aux[i][j+2])
    
    #Land-cells inner channel properties
    data['cellchannel'][:] = list(map(float, aux[-1]))
    
    #Finish
    return data
#---------------------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------------
def parameters_update(**kwargs): #(NoL, cell1, cell2, lktype, params, cellchannel, filename=None):
    """ Update two hydrodynamic model's files: The links' parameters file and the part of the domain properties
    file related to links data. It reads the file, replace the components provided (key:value in kwargs that
    match with the expected inputs) and overwrite the file.
    INPUTS*:
        Same as parameters_write() but 'filename'. This one works as explained below
        filename |str| = if provided will replace the value in fn_['param'] before writing the updated file.
    
    * kwargs is a dictionary storing all "variable_name=value" entries passed to this function
    """
    # Read domain properties file, then the links' parameters one
    data = domain_read()
    data.update(parameters_read())
    
    # Replacing provided components
    for key in ['NoL', 'cell1', 'cell2', 'lktype', 'params', 'cellchannel']:
        if key in kwargs:
            data[key] = kwargs[key]
    
    # Rewrite domain file
    domain_write(**data)
    
    # If filename was provided, replace it in fn_ dictionary
    if 'filename' in kwargs:
        fn_['param'] = kwargs['filename']
        del(kwargs['filename'])
    
    # Write parameters' file
    parameters_write(**data)
    
    # Finish
    return True
#---------------------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------------
def parameters_write(**kwargs): #(NoL, cell1, cell2, lktype, params, cellchannel, filename=None):
    """ Write the hydrodynamic model's file with links' parameters
    INPUTS*:
        NoL |int| = number of links in the domain
        cell1, cell2, lktype |array-like| = sequence of NoL integers of outlet cell's index, inlet cell's index
            and link-type. This function will add 1 to cell1 and cell2 to change to FORTRAN indexation
        params |np.array((NoL,12), float)| = links' parameter matrix
        cellchannel |array-like| = the three values of land-cell-inner-channel parameters: bottom width, lateral
            slope and channel depth
        filename |string| = path to file with links' parameters
    
    * kwargs is a dictionary storing all "variable_name=value" entries passed to this function
    """
    # Open file to write
    if 'filename' in kwargs:
        f = open(kwargs['filename'], 'w')
    else:
        f = open(fn_['param'], 'w')
    
    # For each link ...
    for i in range(kwargs['NoL']):
        #... write the link's cells numbers. Adds 1 to change cell counting from 1 on (instead of 0)
        f.write('%5i %5i' % (kwargs['cell1'][i]+1, kwargs['cell2'][i]+1))
        # and then write the link's parameters, according to its type
        if kwargs['lktype'][i] == 61:
            #land-land link type
            f.write(' %9.3f %9.3f %9.5f\n' % tuple(kwargs['params'][i,0:3]))
        elif kwargs['lktype'][i] == 6:
            #land-river link type
            f.write(' %9.3f %9.3f %9.5f %9.4f %9.4f\n' % tuple(kwargs['params'][i,0:5]))
        elif kwargs['lktype'][i] == 0 or kwargs['lktype'][i] == 10:
            #river-river link type
            f.write(' %9.3f %9.3f %9.5f %9.5f %9.4f %9.4f %9.4f\n' % tuple(kwargs['params'][i,0:7]))
        elif kwargs['lktype'][i] == 1:
            #weir-structure link type without gate
            f.write((11*' %9.4f' + '\n') % tuple(kwargs['params'][i,0:11]))
        elif kwargs['lktype'][i] == 11:
            #weir-structure link type with gate
            f.write((12*' %9.4f' + '\n') % tuple(kwargs['params'][i,0:12]))
            
    # Write the bottom width, lateral slope and channel depth parameters of land-type cells
    f.write(' %6.3f %6.3f %6.3f\n' % tuple(kwargs['cellchannel']))
    
    # End of file, close it
    f.close()
    
    #Finish
    return True
#---------------------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------------
def periods_write(periods, filename, stepNames=None):
    """ Dumps the periods data in a CSV-like file """
    if stepNames is None:
        stepNames = [str('series %i' % (i+1)) for i in range(len(periods))]
    with open(filename, 'w') as f:
        for i in range(len(periods)):
            for j in range(len(periods[i])):
                f.write('%s,cycle %i' % (stepNames[i],j+1))
                for t in periods[i][j]:
                    f.write(',%i' % t)
                f.write('\n')
#---------------------------------------------------------------------------------------------------------------
 
#---------------------------------------------------------------------------------------------------------------
def special_tiles_read(filename=None):
    """ Read data from file with properties of special areas in the domain.
    INPUT:
        filename |string| = special tiles/area file's path/name. If it is none, read from fn_['tiles']
    RETURN:
        tiles |dict| = dictionary with properties of each special area (each area is a dict's key)
            tiles[key]['cells'] = array with cells' number inside the special area
            tiles[key]['roughness'] = Manning's roughness coefficient in the special area
    """
    # Read file content
    if filename is None:
        name = fn_['tiles']
    else:
        name = filename
    with open(name, 'r') as f:
        fc = f.readlines()
    
    # Initialisations
    i, tiles = 0, {}
    
    # Looping through the file
    while True:
        line = fc[i].split()
        tile, Mann, N = line[0], float(line[1]), int(line[2])
        tiles[tile] = {'roughness': Mann}
        cells = list(map(int, fc[i+1:i+N+1]))
        tiles[tile]['cells'] = np.array(cells, dtype=int)
        i = i + N + 1
        if i >= len(fc):
            break
    
    # Finish
    return tiles
#---------------------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------------
def special_tiles_write(tiles, filename=None):
    """ Write tiles name, roughness and selected cells of each special area 
    within tiles dictionary.
    INPUT:
        tiles |dict| = dictionary with properties of each special area (each area is a dict's key)
        tiles[key]['cells'] = array with cells' number inside the special area
        tiles[key]['roughness'] = unique Manning's roughness coefficient in the special area
        filename |string| = special tiles/area file's path/name. If it is None use fn_['tiles'] instead
    RETURN:
        
    """
    # Open file to write
    if filename is None:
        f = open(fn_['tiles'], 'w')
    else:
        f = open(filename, 'w')
    
    # For each special area ...
    for key in tiles.keys():
        #... write the area's name, Manning's roughness coefficient and number of
        #cells in this selection
        f.write('%s   %.4f   %i\n' % (key, tiles[key]['roughness'],
            len(tiles[key]['cells'])))
        # Write the cell's number within the special area
        for i in range(len(tiles[key]['cells'])):
            f.write('%i\n' % tiles[key]['cells'][i])
        
    # End of file, close it
    f.close()
    
    # Finish
    return True
#---------------------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------------
def table_series_read(filename, firstColumnAsIndex=True):
    ''' Use pandas.read_csv to load a DataFrame from file. '''
    if firstColumnAsIndex:
        table = pd.read_csv(filename, index_col=0)
    else:
        table = pd.read_csv(filename)
    return table
#---------------------------------------------------------------------------------------------------------------
#===============================================================================================================





























#===============================================================================================================
# DATA PROCESSING FUNCTIONS
#===============================================================================================================
def define_parameters(**ka): #(params, cell1, cell2, lktype, Mann, x, y, z, dx, dy, botwid, latslp, chndep):
    """ Review link's parameters related to properties of the linked cells
    INPUTS*:
        params |array(NoL,12)| = parameter's matrix for all NoL links
        cell1, cell2 |array(NoL)| = index of first (inlet) and second (outlet) cells in the link, respectively
        lktype |array(NoL)| = link type code
        Mann |array(NoC)| = Manning's roughness coefficient at each NoC cells
        x, y, z, dx, dy |array(NoC)| = arrays of cells' x-coordinate, y-coordinate, bottom elevation (z),
            x-length and y-length
        botwid, latslp, chndep |array(NoC)| arrays of channel properties: bottom width, lateral slope, channel depth
    RETURN:
        Parameters are overwrited in params matrix
    
    * ka is a dictionary storing all "variable_name=value" entries passed to this function
    """
    for i in range(ka['params'].shape[0]):
    
        #Getting the average properties between cells
        c1, c2 = ka['cell1'][i], ka['cell2'][i]
        if ka['x'][c1] == ka['x'][c2]:
            #cells are in the same column
            dist = abs(ka['y'][c1] - ka['y'][c2])
            face_width = 0.5 * (ka['dx'][c1] + ka['dx'][c2])
        elif ka['y'][c1] == ka['y'][c2]:
            #cells are in the same row
            dist = abs(ka['x'][c1] - ka['x'][c2])
            face_width = 0.5 * (ka['dy'][c1] + ka['dy'][c2])
        else:
            msg = str('\n Cells %i and %i from link %i are not aligned\n' %
                      (c1, c2, i))
            raise ValueError(msg)
        
        if ka['lktype'][i] == 61:    #Land-Land link:
            ka['params'][i,0] = dist       #distance between cells
            ka['params'][i,1] = face_width #interface cross-section width
            ka['params'][i,2] = 0.5 * (ka['Mann'][c1] + ka['Mann'][c2]) #surface's roughness
            
        elif ka['lktype'][i] == 6:  #River-Land link:
            ka['params'][i,0] = dist       #distance between cells
            ka['params'][i,1] = face_width #interface cross-section width
            ka['params'][i,2] = 0.5 * (ka['Mann'][c1] + ka['Mann'][c2]) #surface's roughness
            #elevation of land-phase        
            if ka['chndep'][c1] > 0.0:
                #first cell is the river
                ka['params'][i,3] = ka['chndep'][c1] + ka['z'][c1]
                ka['params'][i,4] = ka['z'][c2]
            else:
                #first cell is land type (second is river type)
                ka['params'][i,3] = ka['z'][c1]
                ka['params'][i,4] = ka['chndep'][c2] + ka['z'][c2]
        
        elif ka['lktype'][i] == 0: #River-River link:
            ka['params'][i,0] = dist       #distance between cells
            ka['params'][i,1] = face_width #interface cross-section width
            ka['params'][i,2] = 0.035      #channel-phase roughness
            ka['params'][i,3] = 0.5 * (ka['Mann'][c1] + ka['Mann'][c2]) #land-phase surface roughness
            ka['params'][i,4] = 0.5 * (ka['botwid'][c1] +ka['botwid'][c2])
            ka['params'][i,5] = 0.5 * (ka['latslp'][c1] + ka['latslp'][c2])
            ka['params'][i,6] = 0.5 * (ka['chndep'][c1] + ka['chndep'][c2])
        
        else:
            #Other link types: No changes at all. They should be defined manually
            pass
    
    #Finish
    return True
#---------------------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------------
def define_periods(ndata, record_step, single_duration, skip_first=0, skip_last=0):
    """ Create lists of indexes from a time series to identify time-even intervals where some computation will
    take place later.
    INPUTS:
        ndata |int| = number of records in the output time-series created by the hydrodynamic model
        record_step |int| = time-length (s) between records.
        single_duration |int| = time-length (s) of a single interval.
        skip_first |int| = start search of intervals after this time-length (s) from the beginning of the series.
        skip_last |int| = stop search of intervals before this time-length (s) from the end of the series.
    RETURN:
        periods |list of arrays((*), int)| = list with arrays containing the indexes of the time-series. Each
            array store the indexes for a single interval.
    """
    #time array (seconds) of each record in the series (note that h, v and Q
    #matrices always start at t = Dt, not at t=0)
    tdata = np.arange(1, ndata+1) * record_step
    #selecting those times within the desired range
    ti, = np.where((tdata >= skip_first) & (tdata <= (tdata[-1] - skip_last)))
    #computing number of records per interval
    nri = int(single_duration / record_step)
    #initialize periods list
    periods = []
    
    #select indexes per interval
    for i in range(0, ti.size, nri):
        iN = min(ti.size, i+nri)
        temp = ti[i:iN]
        
        if temp.size > 0.9 * nri:
            periods.append(temp)
    
    return periods
#---------------------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------------
def get_hydrodynamic_id(filename=fn_['setup']):
    ''' Return the 5-characters-long ID of the hydrodynamic simulation, found in the first line of the setup
    file '''
    with open(filename,'r') as f:
        name = f.readline()[1:-2]    #f.readline() returns something like '"playa"\n'
    return name
#---------------------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------------
def relationships(ed):
    ''' Find the related cells and links to each cell in the domain. It create arrays with the indexes for
    neighbouring cells, links and also the flow signal, for each cell.
    INPUTS:
        ed |EGM-data class| = variable from class with at least the following domain's attributes:
            NoC, NoL, cell1, cell2, x and y
    RETURN:
        nbcells |list of np.array((N), dtype=int)| = a list with ed.NoC arrays. Each array stores the index of
    the N neighbouring cells.
        links |list of np.array((M), dtype=int)| = a list with ed.NoC arrays. Each array stores the index of the
    M links (the index of links goes from 0 to ed.NoL-1, while the index of cells goes from 0 to ed.NoC-1)
        signal |list of np.array((M), dtype=int)| = a list with ed.NoC arrays. Each array stores the signal
    (+1 or -1) for each of the M links. If the current cell is in ed.cell1 of the link, a positive
    velocity/discharge means that the flow is going from cell1 to cell2. A negative means that the cell1 is
    receiving the flow from cell2. However, if the current cell is in ed.cell2, than these orientations are the
    opposite.
        same_x, same_y |list with ed.NoC arrays like np.array((P), dtype=int)| = store, for each cell, those
    links with cells in the same row (same_y) or in the same column (same_x). Therefore P = 1 or P = 2
    '''
    nbcells = [[] for i in range(ed.NoC)]
    links   = [[] for i in range(ed.NoC)]
    signal  = [[] for i in range(ed.NoC)]
    
    for i in range(ed.NoL):
        c1 = ed.cell1[i]
        c2 = ed.cell2[i]
        
        #Adding neighbour cells to the list
        nbcells[c1].append(c2)
        nbcells[c2].append(c1)
        
        #Adding links
        links[c1].append(i)
        links[c2].append(i)
        
        #Setting signals: positive for incoming flow, negative for outcome
        signal[c1].append(-1)
        signal[c2].append(1)
    
    # Identifing horizontal and vertical links
    same_x = [[] for i in range(ed.NoC)]
    same_y = [[] for i in range(ed.NoC)]
    
    for i in range(ed.NoC):            
        xi, yi = ed.x[i], ed.y[i]
        
        for j in range(len(nbcells[i])):
            
            nc = nbcells[i][j]
            xj, yj = ed.x[nc], ed.y[nc]
            
            if xi == xj:    #vertical links
                same_x[i].append(links[i][j])
                
            elif yi == yj:    #horizontal links
                same_y[i].append(links[i][j])
                
            else:
                msg = '\n ERROR: In relationships()'
                msg += str('    Cells %i (%.1f,%.1f) and %i (%.1f, %.1f) are not alligned!\n' % (i, xi, yi, nc,
                           xj, yj))
                raise IndexError(msg)
    
    #Converting inner lists to arrays of integer type
    for i in range(ed.NoC):
        nbcells[i] = np.array(nbcells[i], dtype=int)
        links[i] = np.array(links[i], dtype=int)
        signal[i] = np.array(signal[i], dtype=int)
        same_x[i] = np.array(same_x[i], dtype=int)
        same_y[i] = np.array(same_y[i], dtype=int)
    
    return {'nbcells': nbcells, 'links': links, 'signal':signal, 'same_x': same_x, 'same_y':same_y}
#===============================================================================================================





























#===============================================================================================================
# MODELS FUNCTIONS
#===============================================================================================================
def accretion(D, V, C, rule='SE Australia', returnBiomass=False):
    """ Computes the biomass and soil accretion as function of mean depth below high tide, vegetation type and
    average sediment concentration.
    
    INPUTS:
    D |array((NoC), float)| = mean depth below high tides (m) in each cell.
    V |array((NoC), int)| = vegetation type in each cell.
    C |array((NoC), float) = average sediment concentration (g/m3) in each cell.
    rule |string| = identify which set of rules will be applied. Current options: 'SE Australia', 'England'.
        * ATENTION: The same rule should be applied in the vegetation function
    returnBiomass |bool| = True to return biomass productivity and accretion. Otherwise return accretion only.
    
    RETURN:
    A |array((NoC), float)| = array of total accreation (m), in each cell.
    <opt> B |array((NoC), float)| = array of biomass production, in g/m2,
        on each cell.    
    """
    A = np.zeros(D.shape[0], dtype=float)
    B = np.zeros(D.shape[0], dtype=float)
    
    if rule == 'SE Australia':
        #Saltmarsh
        spot = np.where(V == 2)
        B[spot] = 8384.0*D[spot] - 16767*(D[spot]**2)
        A[spot] = C[spot] * (0.00009 + 6.2e-7*B[spot]) * D[spot]
    
        #Mangrove
        spot = np.where(V == 3)
        B[spot] = 7848.9*D[spot] - 6037.6*(D[spot]**2) - 1328.3
        A[spot] = C[spot] * (0.00009 + 1.2e-7*B[spot]) * D[spot]
        
    elif rule == 'England':
        #Saltmarsh
        spot = np.where(V == 2)
        B[spot] = 6400.0*D[spot] - 6400*(D[spot]**2)
        A[spot] = C[spot] * (0.00009 + 6.2e-7*B[spot]) * D[spot]
    
    if returnBiomass:        
        return A, B
    
    else:        
        return A
#---------------------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------------
def add_accretion(z, d, A):
    """ Add the accretion amount 'A' to the domain elevation. In land-type cells, where channel depth is zero,
    add to the cell's bottom elevation 'z', otherwise increase the channel depth 'd'. All inputs are in metres
    with format np.array(NoC, float). """
    z[d == 0] += A[d == 0]
    d[d > 0] += A[d > 0]
    return z, d
#---------------------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------------
def bathtub_depths(wl, elev):
    """ Return a water depths series using the bathtub method
    INPUTS:
        wl |array((N), float)| = series of N records of water-levels
        elev |array((NoC), float)| = array with bottom elevation in a domain with NoC cells.
    RETURN:
        wd |array((N, NoC), float)| = matrix of N records of water-depths for each cell
    """
    wd = wl[:, np.newaxis] - elev
    wd[wd < 0] = 0.0
    return wd
#---------------------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------------
def bathtub_ssc(D, Cmax):
    """ Compute average sediment concentration using the simple linear model fitted for Koorangang Island
    (Hunter Estuary).
    INPUTS:
        D |array((NoC), float)| = array of mean depth below high tides, in metres, in each cell.
        Cmax |float| = sediment concentration at the tide-input-creek
    RETURN:
        C |array((NoC), float)| = array of mean sediment concentration, in g/m3, in each cell.
    """
    C = (0.55 * D + 0.32) * Cmax
    return C
#---------------------------------------------------------------------------------------------------------------
    
#---------------------------------------------------------------------------------------------------------------
def hillslope_diffusion(elev, Dcoef, lines, dx = 10.0):
    """ Apply a simple "geomorphic diffusion" model for landform evolution, in
    which the downhill flow of soil is assumed to be proportional to the
    (downhill) gradient of the land surface multiplied by a transport coefficient.
    INPUTS:
        elev |array((NoC), float)| = array with terrain elevation in the
            simulated domain (NoC = Number of Cells)
        Dcoef |float| = transport rate coefficient (hillslope diffusivity,
            m2/year).
        lines|list of arrays((*),int)| = list with sequence of cells of each
            slope profile.
        dx |float| = distance between cells (m)
    RETURN:
        smel |array((NoC), float)| = smoothed elevation array after diffusion
    
    IMPORTANT: Must pay attention that Dcoef is m2 PER YEAR, thus assure a
        coherent value if applying for a different time-length.
    """
    smel = np.copy(elev)
    if isinstance(dx, float):
        dx = np.ones(len(lines), dtype=float) * dx
    
    for i, sp in enumerate(lines):
        #prepend and append elevation in the first and last cells, respectively,
        #to be able to apply the method at the edges of the profile
        tmp = np.concatenate(([smel[sp[0]]], smel[sp], [smel[sp[-1]]]))
        qs = -Dcoef * np.diff(tmp) / dx[i]
        dzdt = -np.diff(qs)/dx[i]
        smel[sp] += dzdt
        
    return smel
#---------------------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------------
def hydrodynamic_model(execname=fn_['exec'], return_outcome=False, verbose=False):
    """ Run the hydrodynamic model through its executable."""
    #Define command to be execute in a Windows or Unix system
    if os.name == 'nt':
        command = execname
    else:
        command = str('./%s' % execname)
        
    #Addind argument to supress screen outputs if verbose is False
    if not verbose:
        command += ' 1'
        
    #Running the model
    os.system(command)
    
    #Returning data if requested
    if return_outcome:
        hdOut = highFreqData()
        hdOut.load_native_outputs()
        return hdOut
    
    else:
        return True
#---------------------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------------
def tidal_indexes(wd, intervals=None, dLimit=0.10):
    """ Compute the hydroperiod and mean depth below high tide
    INPUTS:
        wd |array((Nrec,NoC), float)| = matrix of water depths. Nrec is the number of records and NoC is the
            number of cells. To make it work for River-Type cells, pass wd-chndep as wd
        intervals |list of arrays((*), int)| = a list of arrays, where each array stores a slice of time indexes
            of the wd-matrix, in a way that wd[intervals[i],c] return the water depths at the cell c for the
            time indexes within intervals[i]. If intervals is None, then the entire series is considered as a
            single interval
        dLimit |float| = value above which a cell is considered inundated (used for the hydroperiod)
    RETURN:
        HP |array((NoC), float)| = array with the values of hydroperiod (%)
        MD |array((NoC), float)| = array with the values of mean depth below high tide (m)
    """
    if intervals is None:
        intervals = [np.arange(wd.shape[0])]
    
    if isinstance(intervals, list):
        if intervals[0] is None:
            intervals = [np.arange(wd.shape[0])]
    
    #Computing the hydroperiod and mean depth below high tide
    HP = np.zeros((wd.shape[1]), dtype=float)
    MD = np.zeros((wd.shape[1]), dtype=float)
    n = 0
    
    for cycle in intervals:
        n += cycle.size
        
        for i in cycle:
            wet = np.where( wd[i,:] >= dLimit )
            HP[wet] += 1.0
            
        for j in range(wd.shape[1]):
            MD[j] += np.max(wd[cycle,j])
        
    HP = HP * 100 / n
    MD = MD / len(intervals)
    MD[MD < 0.0] = 0.0
    
    return HP, MD
#---------------------------------------------------------------------------------------------------------------
#
#
##------------------------------------------------------------------------------
#def replace_run_conditions(bd=None, endtime=None, recstep=None, iwd=None,
#       prev=None):
#    """ Rewrites partially or completelly some setup/input files of the
#    hydrodynamic model, depending on received arguments (None means it will not
#    be changed)
#    INPUTS:
#        bd |dict| = boundary depths dictionary. It must have the same items
#            returned by load_boundary_depths()
#        endtime, recstep |int, int| = total length of the simulation and timestep
#            length for records in the output files. Both in seconds.
#        iwd |array or list| = sequence of floats (must have the same size of
#            number of cells in the domain) with the initial water depths
#        prev |bool| = True to set hydrodynamic run starting from previous state,
#            False to start from given initial conditions.
#    """
#    if bd is not None:
#        #Replace file with water depths at boundary cells
#        file = open(fn_['bounds'], 'w')
#        file.write(' %i  %i  %i\n' % (bd['Ndata'], bd['ts'], bd['Nbounds']))
#        
#        for i in range(bd['Nbounds']):
#            file.write('%i %i %i\n' % tuple(bd['spots'][i,:]))
#            
#            for j in range(bd['Ndata']):
#                file.write('%.6f\n' % bd['series'][i,j])
#        
#        file.close()
#        
#    if (endtime is not None) or (recstep is not None) or (iwd is not None):
#        #Replace information in the initial depths file
#        with open(fn_['init'], 'r') as file:
#            fct = file.readlines()
#            line1 = fct[1].split()
#        
#        #changing end time of simulation and/or recording timestep
#        if endtime is not None:
#            line1[-2] = str(endtime)
#        
#        if recstep is not None:
#            line1[-1] = str(recstep)
#        
#        fct[1] = ' '.join(line1) + '\n'
#        
#        #changing initial water depths
#        if iwd is not None:
#            fct = fct[0:2]
#            fct.append('')
#            
#            for i in range(len(iwd)):
#                if i > 0 and i%10 == 0:
#                    fct[-1] += '\n'
#                    fct.append('')
#                fct[-1] += str(' %.6f' % iwd[i])
#        
#        #rewriting file
#        with open(fn_['init'], 'w') as file:
#            file.writelines(fct)
#        
#    if prev is not None:
#        #Replace option in main setup file
#        with open(fn_['setup'], 'r') as file:
#            fct = file.readlines()
#        
#        if prev:
#            fct[1] = "'S'\n"
#        else:
#            fct[1] = "'N'\n"
#        
#        with open(fn_['setup'], 'w') as file:
#            file.writelines(fct)
#    
#    #concluded
##------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------------
def sediment_transport_model(fco, sscIn, ssc0, h, v, Q, h0, returnSettled=False, printStats=True):
    """ Run the sediment transport model
    INPUTS:
        fco = a framework class object containing the properties: fco.NoC, fco.dx, fco.dy, fco.Ucd, fco.ws,
            fco.input_cells and fco.Dt_HST. In short, an initialised EGMframework object.
        sscIn |array(NoTS)| = array with boundary condition for sediment concentration over 'NoTS' time steps as
            in the outputs of the hydrodynamic model.
        ssc0 |array(NoC)| = array with initial condition for sediment concentration in the 'NoC' cells of the
            domain
        h |array(NoTS,NoC)| = water depths in 'NoC' cells for each 'NoTS' time steps
        v, Q |array(NoTS,NoL)| = water velocity and water discharge in 'NoL' links for each 'NoTS' time steps
        h0 |array(NoC)| = array with initial condition for water depth in the 'NoC' cells of the domain
        returnSettled |bool| = True to return the settled sediment mass as well
        printStats |bool| = True to print in the screen the stats of model convergence
    RETURN:
        *settled |array(nt,NoC)| = array with mass of settled sediment. Returned only if returnSettled == True
        ssc |array(nt,NoC)| = array of suspended sediment concentration
    """
    if printStats:
        print('\n Running Sediment Transport Model ...')
    
    info = [[], []]

    # Constants
    gama = fco.Dt_HST / (fco.dx * fco.dy)    # = time step [s] / area [m^2]
    
    # Initializing sediment concentration matrix, ssc
    ssc = np.zeros(h.shape, dtype=float)
    if returnSettled:
        settled = np.zeros(h.shape, dtype=float)    #settled sediment [g/m2]
    
    # Auxiliar matrices of sediment concentration (used to solve the model numerically)
    Cst, Cnd = np.zeros(fco.NoC, dtype=float), np.zeros(fco.NoC, dtype=float)

    # Other matrices of the model
    u = np.zeros(fco.NoC, dtype=float)       #average water velocity [m/s]
    Pd = np.zeros(fco.NoC, dtype=float)      #probability of deposition [adim.]
    den = np.zeros(fco.NoC, dtype=float)     #denominator of the formula to find C_(i)^(t+Dt)
    h_prev = np.zeros(fco.NoC, dtype=float)  #water depth in t-1
    ssc_prev = np.zeros(fco.NoC, dtype=float)  #ssc in t-1
    
    # COMPUTING SEDIMENT CONCENTRATION OVER TIME
    #-----------------------------------------------------------------------------------------------------------
    for t in range(h.shape[0]):

        #Set initial guess of future concentration as the values in the past time step
        if t == 0:
            Cst[:] = ssc0[:]
            Cnd[:] = ssc0[:]
        else:
            Cst[:] = ssc[t-1,:]
            Cnd[:] = ssc[t-1,:]
            
        #Concentration in the input cells are known
        Cst[fco.input_cells] = sscIn[t]
        Cnd[fco.input_cells] = sscIn[t]

        #Other initialisations
        Pd[:] = 0.0    # deposition probability
        flows = []     # discharges in/out the cell

        #Computing variables that does not change in the iteration process
        for c in range(fco.NoC):
            
            # Magnitude of cell's velocity vector
            if len(fco.same_y[c]) == 0:
                vel_x = 0.0
            else:
                vel_x = np.average(v[t,fco.same_y[c]])
            if len(fco.same_x[c]) == 0:
                vel_y = 0.0
            else:
                vel_y = np.average(v[t,fco.same_x[c]])
            u[c] = (vel_x**2 + vel_y**2)**0.5

            # Probability (actually, a percentage) of deposition
            if u[c] < fco.Ucd:
                Pd[c] = 1 - (u[c]/fco.Ucd)**2

            # Discharges in/out the cell
            qc = Q[t,fco.links[c]] * fco.signal[c]
            flows.append(qc)

            # Sum of negative discharges
            negSum = np.sum(qc[qc<0])

            # Computing the denominator of the formula to find C_(c)^(t)
            den[c] = h[t,c] - fco.Dt_HST * Pd[c] * fco.ws - gama[c] * negSum
        
        # Inicial values for maximum error and iteration number
        maxError, it = 9.e9, 0
        
        if t == 0:
            h_prev[:] = h0[:]
            ssc_prev[:] = ssc0[:]
        else:
            h_prev[:] = h[t-1,:]
            ssc_prev[:] = ssc[t-1,:]
        
        # Iterating until reach the error tolerance of 0.05 g/m3
        while(maxError > 0.049):
            
            for c in range(fco.NoC):
                
                if fco.input_cells[c]:
                    #do not compute the sediment concentration at input cells
                    continue
                
                if abs(den[c]) < 0.000001:
                    #do not compute the ssc where water-depth is too low
                    Cnd[c] = 0.0
                    continue

                posSum = 0.0    #sum of (discharge x concentration) of incoming cells    
                for j in range(flows[c].size):
                    if flows[c][j] > 0:    
                        posSum += flows[c][j] * Cst[fco.nbcells[c][j]]

                # Computing new guess for sediment concentration
                Cnd[c] = (h_prev[c] * ssc_prev[c] + gama[c] * posSum) / den[c]

            maxError = np.max(np.absolute(Cnd - Cst))
            it += 1
            Cst[:] = Cnd[:]
            
        # Passing values of concentration
        ssc[t,:] = Cnd[:]
        info[0].append(maxError)
        info[1].append(float(it))

        # Settled sediment [g/m2]
        if returnSettled:
            settled[t,:] = fco.Dt_HST * Pd[:] * (-1 * fco.ws) * ssc[t,:]
    #-----------------------------------------------------------------------------------------------------------
    
    # Model running efficiency
    if printStats:
        avgE = sum(info[0]) / len(info[0])
        avgN = sum(info[1]) / len(info[1])
        msg = ' > [avg Error, max Error, avg Nit, max Nit] ='
        print(msg, str('[%f, %f, %f, %f]' % ( avgE, max(info[0]), avgN, max(info[1]) )) )

    # Returning calculated data
    if returnSettled:
        return ssc, settled

    else:            
        return ssc
#---------------------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------------
def sinusoid(amplitude, mean_level=0.0, wave_length=43200, time_step=300, final_time=86400):
    """ Construct a sinusoidal water level series
    INPUTS:
        amplitude |number| = wave amplitude (m), the difference between top and bottom levels.
        mean_level |number| = constant-value to be added to the sinusoidal wave
        wave_length |integer| = wave length (seconds). Time length between two consecutives crests/valleys
        time_step |integer| = interval between records (seconds) in the series array
        final_time |integer| = time (in seconds) of the last record
    RETURN:
        steps |array| = array with time of each record in the water level series, in seconds.
        levels |array| = array water level records between t=0 and t=wave_length*repeat evenly spaced in
            time_step seconds.
    """
    steps = np.arange(0, final_time + 1, time_step)
    conv = 2 * np.pi / wave_length
    levels = np.sin(steps * conv) * (amplitude/2) + mean_level
    return steps, levels
#---------------------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------------
def SLR(year):
    """ Return the mean sea-level rise for a given year (number or array). """
    return 0.000045982 * year**2 - 0.181198214 * year + 178.471071429
#---------------------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------------
def vegetation_model(HP, MD, rule='SE Australia', no_veg_cells=None):
    """ Applies the vegetation establishment rules due to the tidal indexes in
    each cell.    
    INPUTS:
        HP |array((NoC), float)| = array of hydroperiod (%). NoC = number of cells
        MD |array((NoC), float)| = array of mean depth below high tide (m)
        rule |string| = identify which set of rules will be applied. Current
             options: 'SE Australia', 'England'.
        * ATENTION: The same rule should be applied in the accretion function
        no_veg_cells |array((NoC), boolean)| = True for those cells where
            vegetation must receive code 0
    RETURN:
        V |array((NoC), int)| = array with vegetation-id. These values are
            presented below accordingly to the applied rule:
        > 'SE Australia':
            0 = No vegetation
            1 = Freshwater vegetation
            2 = Saltmarsh
            3 = Mangrove
        > 'England':
            0 = No vegetation
            1 = Freshwater vegetation
            2 = Saltmarsh
    """
    NoC = HP.shape[0]
    V = np.zeros(NoC, dtype=int)
    
    if rule == 'SE Australia':
        for i in range(NoC):
            
            if HP[i] <= 80. and MD[i] <= 0.25:    #Saltmarsh
                V[i] = 2
                
            if HP[i] == 0.0:                      #Freshwater vegetation
                V[i] = 1
            
            if 10. <= HP[i] <= 50.:               #Mangrove
                if MD[i] >= 0.20:
                    V[i] = 3
    
    elif rule == 'England':
        for i in range(NoC):
            
            if MD[i] <= 0.5 and HP[i] <= 80.0:    #Saltmarsh
                V[i] = 2
            
            if HP[i] == 0:                        #Freshwater vegetation
                V[i] = 1
    
    if no_veg_cells is not None:
        V[no_veg_cells] = 0
    
    return V
#===============================================================================================================





























#===============================================================================================================
# PLOTTING FUNCTIONS
#===============================================================================================================
def equalise_yrange(axes, ymin=None, ymax=None):
    """Apply the same y-axis limits to each axis in axes"""
    lims = [9.e9, -9.e9]
    
    for ax in axes.flatten():
        thisRange = ax.get_ylim()
        if thisRange[0] < lims[0]:
            lims[0] = thisRange[0]
        if thisRange[1] > lims[1]:
            lims[1] = thisRange[1]
        
    if ymin is not None:
        lims[0] = ymin
    if ymax is not None:
        lims[1] = ymax
    
    for ax in axes.flatten():
        ax.set_ylim(lims[0], lims[1])

def nan_and_reshape(vector, nan_cells, shape):
    aux = np.copy(vector)
    aux[nan_cells] = np.nan
    return aux.reshape(shape)


def nan_and_reshape_2d(matin, nan_cells, shape3d):
    matout, aux = np.zeros(shape3d), np.copy(matin)
    shape = (shape3d[1], shape3d[2])
    for i in range(shape3d[0]):
        aux[i,nan_cells] = np.nan
        matout[i] = aux[i].reshape(shape)
    return matout


def load_to_plot(framework_output_file, excluding_area=True):
    """ Load data from an EGM run and reshape to grid-like format """
    
    #loading framework outputs, fwo
    with open(framework_output_file, 'rb') as file:
        fwo = pickle.load(file)
    
    if excluding_area:
        #identifing cells to set as NaN values
        cp = fwo.exclude.flatten()
    else:
        #won't exclude anything
        cp = np.zeros(fwo.NoC, dtype=bool)
    
    data = {'NoC': fwo.NoC, 'stages': fwo.stages, 'rule': fwo.rule,
            'shape': (fwo.Nstages, fwo.nrows, fwo.ncols)}
    
    #cell's centre x,y coordinates
    shp = (fwo.nrows, fwo.ncols)
    data['xc'] = nan_and_reshape(fwo.x, cp, shp)
    data['yc'] = nan_and_reshape(fwo.y, cp, shp)
    data['z0'] = nan_and_reshape(fwo.z0, cp, shp)
    
    #cell's face x,y coordinates
    aux1 = fwo.x.reshape(shp) - 0.5*fwo.dx.reshape(shp)
    aux2 = fwo.x.reshape(shp) + 0.5*fwo.dx.reshape(shp)
    xf = np.unique( np.concatenate((aux1, aux2)) )
    aux1 = fwo.y.reshape(shp) - 0.5*fwo.dy.reshape(shp)
    aux2 = fwo.y.reshape(shp) + 0.5*fwo.dy.reshape(shp)
    yf = np.unique( np.concatenate((aux1, aux2)) )
    data['xf'], data['yf'] = np.meshgrid(xf, np.flip(yf))
    
    #eco-geomorphological variables
    data['H'] = nan_and_reshape_2d(fwo.H, cp, data['shape'])
    data['D'] = nan_and_reshape_2d(fwo.D, cp, data['shape'])
    data['V'] = nan_and_reshape_2d(fwo.V.astype(float), cp, data['shape'])
    data['C'] = nan_and_reshape_2d(fwo.C, cp, data['shape'])
    data['A'] = nan_and_reshape_2d(fwo.A, cp, data['shape'])
    data['EG'] = nan_and_reshape_2d(fwo.EG, cp, data['shape'])
    
    #post-processing variables
    data['acA'] = np.zeros(data['shape']) * np.nan
    data['zc'] = np.zeros(data['shape']) * np.nan
    
    for i in range(fwo.Nstages):
        
        if i == 0:
            data['acA'][0] = data['A'][0,:,:]
            data['zc'][0] = nan_and_reshape(fwo.z0, cp, shp)            
        else:
            data['acA'][i] = data['acA'][i-1,:,:] + data['A'][i,:,:]
        
        data['zc'][i] = data['z0'][:,:] + data['EG'][i,:,:]
    
    return data


def plot_vegetation(axis, xgrid, ygrid, veg, rule):
    """ Default plotting of EGM vegetation data
    INPUTS:
        axis |matplotlib.pyplo.Axes object| = axis where the vegetation map will be plotted
        xgrid |array((Nrows+1, Ncols+1), float)| = cell's corners x-coordinates
        ygrid |array((Nrows+1, Ncols+1), float)| = cell's corners y-coordinates
        veg |array((Nrows, Ncols), float)| = EGM vegetation data
        rule |str| = rule applied in the EGM simulation for vegetation/accretion models
        return_legend |bool| = decide if return hand
    RETURN:
        veg_patches = list with rectangular patches to be used in pyplot.legend(handles=veg_patches)
    """
    # Preparing plot
    V, params = reset_vegetation(veg, rule)
    p = axis.pcolormesh(xgrid, ygrid, V, cmap=params['cmap'], norm=params['norm'])
    axis.set_xlim(0,None)
    axis.grid()
    
    # Preparing handles for legend
    veg_patches = []
    for i in range(params['cmap'].N):
        p = mpatches.Patch(color=params['cmap'].colors[i], label=params['label'][i+1])
        veg_patches.append(p)
    return veg_patches


def reset_vegetation(veg, rule='SE Australia'):
    """ Review vegetation code number for pcolormesh plot and return a
    well-suited selection of colormap and colorbar ticks for the applied rule.
    INPUTS:
        veg |array| = matrix of vegetation's type-code, of any shape and dtype
        rule |string| = set of rules applied in the vegetation model
    RETURN:
        reveg |arrays(veg.shape)| = matrix of vegetation with type-code reviewed
            for the pcolormesh plot
        params |dict|:
            params['cmap'] = matplotlib colormap for the current rule
            params['norm'] = matplotlib norm for the current rule
            params['posit'] = ticks position for colorbar
            params['label'] = ticks labels for colorbar
    """
    reveg = np.copy(veg.astype(float))
    
    if rule == 'SE Australia':
        
        a = np.where(reveg == 3.0)
        b = np.where(reveg == 1.0)
        reveg[a] = 1
        reveg[b] = 3
        
        params = {
            'posit': [0, 1, 2, 3, 4],
            'label': ['', 'No Vegetation', 'Mangrove', 'Saltmarsh', 'Freshwater Veg.'],
            'cmap': None,
            'norm': None
        }
        params['cmap'], params['norm'] = mcolors.from_levels_and_colors(
            params['posit'], ['#346fa3', '#6c8061', '#daa60b', '#aade87'],
            extend='neither')
        
    elif rule == 'England':
        
        a = np.where(reveg == 2.0)
        b = np.where(reveg == 1.0)
        reveg[a] = 1
        reveg[b] = 2
        
        params = {
            'posit': [0,1,2,3],
            'label': ['', 'No Vegetation', 'Saltmarsh', 'Freshwater Veg.'],
            'cmap': None,
            'norm': None
        }
        params['cmap'], params['norm'] = mcolors.from_levels_and_colors(
            params['posit'], ['#346fa3', '#daa60b', '#aade87'], extend='neither')
    
    return reveg, params
#===============================================================================================================







if __name__ == "__main__":    
    pass
