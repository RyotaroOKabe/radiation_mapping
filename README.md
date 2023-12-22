# Tetris-inspired detector with neural network for radiation mapping
In recent years, radiation mapping has attracted widespread research attention and increased public concerns on environmental monitoring. In terms of both materials and their configurations, radiation detectors have been developed to locate the directions and positions of the radiation sources. In this process, algorithm is essential in converting detector signals to radiation source information. However, due to the complex mechanisms of radiation-matter interaction and the current limitation of data collection, high-performance, low-cost radiation mapping is still challenging. Here we present a computational framework using Tetris-inspired detector pixels and machine learning for radiation mapping. Using inter-pixel padding to increase the contrast between pixels and neural network to analyze the detector readings, a detector with as few as four pixels can achieve high-resolution directional mapping. By further imposing Maximum a Posteriori (MAP) with a moving detector, further radiation position localization is achieved. Non-square, Tetris-shaped detector can further improve performance beyond the conventional grid-shaped detector. Our framework offers a new avenue for high quality radiation mapping with least number of detector pixels possible, and is anticipated to be capable to deploy for real-world radiation detection with moderate validation.   

[[Paper]](https://arxiv.org/abs/2302.07099)   


<p align="center">
  <img src="assets/SupplementaryMovie01.gif" width="500">
</p>


# Starting OpenMC
[1] Create conda environment for openmc simulations (Linux/Mac)
reference: https://docs.openmc.org/en/stable/quickinstall.html

```
$ conda create --name openmc-env python=3.9.9
$ conda activate openmc-env
$ conda install openmc -c conda-forge

```

You can also do the installation with the following commands:
```
$ conda create -n openmc-env openmc python==3.9.9
$ conda activate openmc-env  
```

   
other libraries
```
scikit-learn==1.2.0
torch==1.11.0
tensorboardX==2.5
torchvision==0.12.0
protobuf==3.19.4
pyemd==0.5.1
imageio==2.19.2
scipy==1.11.4
numpy==1.26.2
matplotlib==3.8.2
```

[2] Download the OpenMC Library

```
$ wget -c https://anl.box.com/shared/static/d359skd2w6wrm86om2997a1bxgigc8pu.xz
$ tar -xf d359skd2w6wrm86om2997a1bxgigc8pu.xz
$ mv mcnp_endfb71/ openmc_library/ 
```

[3] Set the path of `crosssections.xml` file  
Assign the path to `crosssections.xml` by setting `env_config.py`.   
Make a copy of the `env_config_temprate.py` file and rename it to `env_config.py`. Modify the following environment variables in `env_config.py`.   
```  
os.environ['CUDA_VISIBLE_DEVICES']="path to the folder that contains this repo/radiation_mapping/openmc_library/cross_sections.xml" 
```  

You can also specify the path by adding the following line to .bashrc or .profile. 
```
export OPENMC_CROSS_SECTIONS="path to the folder that contains this repo/radiation_mapping/openmc_library/cross_sections.xml" 
```   



# radiation_mapping workflow

We use the folders below for storing files:
**Training data**: ./save/openmc_data/
**Filter layers**: ./save/openmc_data/
**Models**: ./save/openmc_data/
**Intermediate data for radiation mapping**: ./save/mapping_data/
**Output for radiation mapping**: ./save/radiation_mapping/

MAke sure to check the file/folder names for loading/saving outputs when you run each program. If you want to skip MC simulation or training model, you can use our data/models below. You can copy the file/folder from the folder 'saved_files' to the path shown above.   

Detector | MC data | MC filter | Model | Epochs | Note
----- | --- | --- | --- |--- |----- 
2x2 square | sq2_1_data/ | sq2_1_filter/ | sq2_1_model.pt | 200 | 1 source.
S-shape | s_1_data/ | s_1_filter/ | s_1_model.pt | 200 | 1 source.
J-shape | j_1_data/ | j_1_filter/ | j_1_model.pt | 200 | 1 source.
T-shape | t_1_data/ | t_1_filter/ | t_1_model.pt | 200 | 1 source.
10x10 square | sq10_2_data/ | sq10_1_filter/ | sq10_2_model.pt | 200 | 2 sources.
5x5 square | sq5_2_data/ | sq5_1_filter/ | sq5_2_model.pt | 200 | 2 sources.


## MC simulation to get training data (env=openmc-train)
```
$ python gen_data_tetris.py (or gen_data_square.py)
```

## MC simulation to get Filtering Layer (env=openmc-train)
```
$ python gen_filter_tetris.py (or gen_filter_square.py)
```

## Training (env=openmc-train)
```
$ python train_model.py
```

## Simulation with a moving detector (env=openmc-train)
```
$ python run_detector.py
```

## Mapping
<!-- ```
# If you use Drake  
$ radiation_mapping_drake.py  
  
# Else   -->
<!-- If you come across 'NaN' in the output maps, please adjust the value 'factor1' in utils/mapping.py. -->
```
$ python radiation_mapping.py  
```

