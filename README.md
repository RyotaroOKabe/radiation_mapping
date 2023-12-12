# Tetris-inspired detector with neural network for radiation mapping

![](https://github.com/RyotaroOKabe/radiation_mapping/assets/move_detector_S.gif)

# Starting OpenMC

reference: https://docs.openmc.org/en/stable/quickinstall.html

[1] Create conda environment for openmc simulations
```
$ conda create -n openmc-env openmc python==3.9.9
$ conda activate openmc-env  
```

.. or create from the exported env
```
$ conda env create -f openmc-train.yml
```
[2] Download the OpenMC Library

```
$ wget -c https://anl.box.com/shared/static/d359skd2w6wrm86om2997a1bxgigc8pu.xz
$ tar -xf d359skd2w6wrm86om2997a1bxgigc8pu.xz
$ mv mcnp_endfb71/ openmc_library/ 
```

[3] Set the path of crosssections.xml file  
In .bashrc or .profile, add the line shown below:  
```
OPENMC_CROSS_SECTIONS="/The directory where the source code is located/radiation_mapping/openmc_library/cross_sections.xml" 
```   
(temporary) You can assign the path to crosssections.xml by setting env_config.py. You can duplicate the file env_config_template.py, rename it to env_config.py, and assign the path to OPENMC_CROSS_SECTIONS. 

# Other env setting (If you use Drake for MAP analysis)
You need to set up another conda env to run mapping_*.py program
```
$ conda env create -f radmap.yml
```

# radiation_mapping (2D)

## MC simulation to get training data (env=openmc-train)
```
$ gen_filter_tetris.py (or gen_filter_square.py)
```

## MC simulation to get Filtering Layer (env=openmc-train)
```
$ gen_filter_tetris.py (or gen_filter_square.py)
```

## Training (env=openmc-train)
```
$ train_model.py
```

## Simulation with a moving detector (env=openmc-train)
```
$ run_detector.py
```

## Mapping (env=radmap)
```
# If you use Drake  
$ radiation_mapping_drake.py  
  
# Else  
$ radiation_mapping.py  
```

