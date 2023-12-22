# -*- coding: utf-8 -*-
from utils.mapping import mapping, gen_gif

fig_header = 'sq2_1'   # The name of the data that is same as the one in 'run-detector..py'
th_level = 0.7  # Threshold level of the map
map_horiz = [-15,15,30]     # map geometry (horizontal) [m] *Set the same values as 'run_detector.py'.
map_vert = [-5,25,30]   # map geometry (vertical) [m] *Set the same values as 'run_detector.py'.
acolors =['#77AE51', '#8851AE'] # arrow colors (moving direction, front side)
recordpath = f'./save/mapping_data/{fig_header}'
fig_folder = f'./save/radiation_mapping/{fig_header}_{th_level}'

mapping(fig_folder, fig_header, recordpath, 
        map_geometry = [map_horiz, map_vert], 
        threshold=th_level, colors=acolors)   
gen_gif(fig_folder)

