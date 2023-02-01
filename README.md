# radiation_mapping

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


# Other env setting
You need to set up another conda env to run mapping_*.py program
```
$ conda env create -f radmap.yml
```

# radiation_mapping (2D)

## MC simulation to get training data (env=openmc-train)
gen_data_square_v1.py (or gen_data_tetris_v1.py)

## MC simulation to get Filtering Layer (env=openmc-train)
gen_filter_square_v1.py (or gen_filter_tetris_v1.py)

## Training (env=openmc-train)
train_model.py

## Simulation with a moving detector (env=openmc-train)
run_detector.py

## Mapping (env=radmap)
radiation_mapping.py

