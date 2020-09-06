# -*- coding: utf-8 -*-
"""
Test if functions in egmlib are working properly
"""
from egmlib import *

os.chdir('C:\\Wetlands\\Tutorials\\tests')
print()

#---------------------------------------------------------------------------------------------------------------
# Test Set 01: Manipulating HxT boundary file
# to start this test, get a new contar.dat file
#---------------------------------------------------------------------------------------------------------------
data = boundary_depths_read()
print('Test 01A: Calling boundary_depths_read() . . . . . . . . . . . . . . . . . . . . . . . . . . . . . okay')

boundary_depths_write(data['Dt_HT'], data['loc_HT'], data['HT'])
print('Test 01B: Calling boundary_depths_write(Dt_HT, loc_HT, HT) . . . . . . . . . . . . . . . . . . . . okay')

boundary_depths_write(data['Dt_HT'], data['loc_HT'], data['HT'], filename='xcontar.dat')

data['Dt_HT'] = 1000
data['loc_HT'] = data['loc_HT'][0:3,:]
boundary_depths_update(Dt_HT=data['Dt_HT'], loc_HT=data['loc_HT'], filename='xcontar.dat')
new = boundary_depths_read(return_as_matrix=True)
if new['nun_HT'] == 3 and new['Dt_HT'] == data['Dt_HT']:
    print('Test 01C: Calling boundary_depths_update(Dt_HT, loc_HT) . . . . . . . . . . . . . . . . . . . . . .okay')
else:
    print('Test 01C: Calling boundary_depths_update(Dt_HT, loc_HT) . . . . . . . . . . . . . . . . . . . . . .FAILED')

data['HT'] = data['HT'].iloc[0:100,0:3]
boundary_depths_update(HT=data['HT'], filename='contar.dat')
new = boundary_depths_read(return_as_matrix=True)
if new['Nr_HT'] == 100 and new['HT'][0,0] == data['HT'].iloc[0,0] and new['HT'][-1,-1] == data['HT'].iloc[-1,-1]:
    print('Test 01D: Calling boundary_depths_update(HT) . . . . . . . . . . . . . . . . . . . . . . . . . . . okay')
else:
    print('Test 01D: Calling boundary_depths_update(HT) . . . . . . . . . . . . . . . . . . . . . . . . . . . FAILED')

aux_loc_HT = np.array([[1,2,0], [2,3,0]])
aux_levels = np.arange(1, 11) * 10
aux_elev = np.array([np.nan, 20., 40.])
boundary_depths_update(loc_HT=aux_loc_HT, levels=aux_levels, elevs=aux_elev)
new = boundary_depths_read(return_as_matrix=True)
if new['HT'][0,0] == 0 and new['HT'][9,0] == 80. and new['HT'][5,1] == 20.:
    print('Test 01E: Calling boundary_depths_update(levels, elevs) . . . . . . . . . . . . . . . . . . . . . .okay')
else:
    print('Test 01E: Calling boundary_depths_update(levels, elevs) . . . . . . . . . . . . . . . . . . . . . .FAILED')
    print(new['HT'])
print()
#---------------------------------------------------------------------------------------------------------------


#---------------------------------------------------------------------------------------------------------------
# Test Set 02: Manipulating runtime and initial condition file
# to start this test get a new inicial.dat file
#---------------------------------------------------------------------------------------------------------------
data = initial_condition_read()
print('Test 02A: Calling initial_condition_read() . . . . . . . . . . . . . . . . . . . . . . . . . . . . okay')

initial_condition_write(data['ext_inp'], data['Dt_solver'], data['t_final'], data['Dt_output'], data['h0'])
print('Test 02B: Calling initial_condition_write(ext_inp, Dt_solver, t_final, Dt_output, h0) . . . . . . .okay')

initial_condition_write(data['ext_inp'], data['Dt_solver'], data['t_final'], data['Dt_output'], data['h0'],
                        filename='xinicial.dat')

data['ext_inp'] = (0,0)
data['Dt_solver'] = (1,2,3,4)
data['t_final'] = (100,200,302,462)
data['Dt_output'] = 50
data['h0'] = np.arange(1, 103) * 10
initial_condition_update(ext_inp=data['ext_inp'], Dt_solver=data['Dt_solver'], t_final=data['t_final'],
                         Dt_output=data['Dt_output'], h0=data['h0'], filename='xinicial.dat')
new = initial_condition_read()
if new['ext_inp'][1] == data['ext_inp'][1] and new['Dt_solver'][-1] == data['Dt_solver'][-1] and \
    new['t_final'][-1] == data['t_final'][-1] and new['Dt_output'] == data['Dt_output'] and \
    new['h0'][0] == data['h0'][0] and new['h0'][-1] == data['h0'][-1]:
    print('Test 02C: Calling initial_condtion_update(**kwargs) . . . . . . . . . . . . . . . . . . . . . . . .okay')
else:
    print('Test 02C: Calling initial_condtion_update(**kwargs) . . . . . . . . . . . . . . . . . . . . . . . .FAILED')
    print(new)
print()
#---------------------------------------------------------------------------------------------------------------


#---------------------------------------------------------------------------------------------------------------
# Test Set 03: Manipulating domain properties and link type file
# to start this test get a new vincul.dat file
#---------------------------------------------------------------------------------------------------------------
data = domain_read()
print('Test 03A: Calling domain_read() . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .okay')

domain_write(**data)
print('Test 03B: Calling domain_write(**kwargs) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . okay')

domain_write(filename='xvincul.dat', **data)

data['NoC'], data['NoL'] = 3, 6
data['gauge'] = [100, 200, 300, 400, 500, 600]
domain_update(NoC=data['NoC'], NoL=data['NoL'], gauge=data['gauge'], filename='xvincul.dat')
new = domain_read()
if len(new['number']) == data['NoC'] and len(new['gauge']) == len(data['gauge']):
    print('Test 03C: Calling domain_update(NoC, NoL, gauge) . . . . . . . . . . . . . . . . . . . . . . . . . okay')
else:
    print('Test 03C: Calling domain_update(NoC, NoL, gauge) . . . . . . . . . . . . . . . . . . . . . . . . . FAILED')
    print(len(new['number']), len(new['gauge']))

data['x'][:] = 9
data['cell1'][:] = 99
domain_update(x=data['x'], cell1=data['cell1'], cell2=np.arange(1000), lktype=np.zeros(50)+15, \
    filename='vincul.dat')
new = domain_read()
if (new['x'][0] == 9) and (new['x'][1] == 9) and (new['cell1'][2] != 99) and \
    (len(new['cell2']) == len(new['cell1'])) and (new['lktype'][-1] != 15):
    print('Test 03D: Calling domain_update(x, cell1, cell2, lktype, filename) . . . . . . . . . . . . . . . . okay')
else:
    print('Test 03D: Calling domain_update(x, cell1, cell2, lktype, filename) . . . . . . . . . . . . . . . . FAILED')
print()
#---------------------------------------------------------------------------------------------------------------


#---------------------------------------------------------------------------------------------------------------
# Test Set 04: Manipulating links' parameters and/or the part of domain properties related to links' data
# to start this test get a new pair of vincul.dat and param.dat files and rename the first to vincul2param.dat
#---------------------------------------------------------------------------------------------------------------
fn_['domain'] = 'vincul2param.dat'
data = parameters_read()
print('Test 04A: Calling parameters_read() . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .okay')

parameters_write(**data)
print('Test 04B: Calling parameters_write(**kwargs) . . . . . . . . . . . . . . . . . . . . . . . . . . . okay')

parameters_write(filename='xparam.dat', **data)

old = deepcopy(data.copy())
data['NoL'], data['NoC'] = 6, 3
data['lktype'][0] = 10
data['params'][0,:] = 55555.55555
parameters_update(**data)
new = parameters_read()
if (new['NoL'] == 6) and (new['params'][0,-1] == 0) and (new['params'][0,4] > 50000):
    print('Test 04C: Calling parameters_update(NoC, NoL, lktype, params) . . . . . . . . . . . . . . . . . . .okay')
else:
    print('Test 04C: Calling parameters_update(NoC, NoL, lktype, params) . . . . . . . . . . . . . . . . . . .FAILED')

data['cell1'][:] = 1001
data['cell2'] = np.arange(2001,2100)
parameters_update(cell1=data['cell1'], cell2=data['cell2'], filename='xparam.dat')
new = parameters_read()
if (new['cell1'][-1] == 1001) and (new['cell2'][0] == 2001) and (len(new['cell2']) == data['NoL']):
    print('Test 04D: Calling parameters_update(cell1, cell2, filename) . . . . . . . . . . . . . . . . . . . .okay')
else:
    print('Test 04D: Calling parameters_update(cell1, cell2, filename) . . . . . . . . . . . . . . . . . . . .FAILED')
print()
#---------------------------------------------------------------------------------------------------------------


#---------------------------------------------------------------------------------------------------------------
# Test Set 05: Reading and writing special tiles file
# to start this test get a new special_tiles.dat
#---------------------------------------------------------------------------------------------------------------
data = special_tiles_read()
print('Test 05A: Calling special_tiles_read() . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . okay')

special_tiles_write(data)
print('Test 05B: Calling special_tiles_write(tiles) . . . . . . . . . . . . . . . . . . . . . . . . . . . okay')

data['TIDE']['cells'] = np.arange(12)
data['TIDE']['roughness'] = 0.99999
special_tiles_write(filename='xtiles.dat', tiles=data)
new = special_tiles_read('xtiles.dat')
if (len(new['TIDE']['cells']) == 12) and (data['TIDE']['roughness'] > 0.99):
    print('Test 05C: Calling special_tiles_write(filename, tiles) . . . . . . . . . . . . . . . . . . . . . . okay')
else:
    print('Test 05C: Calling special_tiles_write(filename, tiles) . . . . . . . . . . . . . . . . . . . . . . FAILED')
print()
#---------------------------------------------------------------------------------------------------------------


#---------------------------------------------------------------------------------------------------------------
# Test Set 06: Reading and writing hillslope profiles file
# to start this test get a new profiles.dat
#---------------------------------------------------------------------------------------------------------------
data, delta = hillslope_profiles_read()
print('Test 06A: Calling hillslope_profiles_read() . . . . . . . . . . . . . . . . . . . . . . . . . . . .okay')

hillslope_profiles_write(data, delta)
print('Test 06B: Calling hillslope_profiles_write(tiles) . . . . . . . . . . . . . . . . . . . . . . . . .okay')

data = [np.arange(10), np.arange(10,20), np.arange(20,30)]
delta = 500
hillslope_profiles_write(profiles=data, delta_x=delta, filename='xprofiles.dat')
profs, d = hillslope_profiles_read('xprofiles.dat')
if (len(profs) == 3) and (len(profs[2]) == 10) and (d[-1] == 500):
    print('Test 06C: Calling hillslope_profiles_write(profiles, delta_x, filename) . . . . . . . . . . . . . .okay')
else:
    print('Test 06C: Calling hillslope_profiles_write(profiles, delta_x, filename) . . . . . . . . . . . . . .FAILED')
print()
#---------------------------------------------------------------------------------------------------------------
