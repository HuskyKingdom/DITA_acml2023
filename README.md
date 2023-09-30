# Learning to Terminate in Object Navigation

PyTorch implementation of our ACML 2023 paper **Learning to Terminate in Object Navigation** in AI2-THOR environment. This implementation is modified based on [SAVN](https://github.com/allenai/savn) and [MJOLNIR_O](https://github.com/cassieqiuyd/MJOLNIR). Please refer to [our paper](https://arxiv.org/pdf/2309.16164v1.pdf) for more details. 



![DITA Visualization Demo](https://raw.githubusercontent.com/HuskyKingdom/DITA_corl2023/main/demo.gif)


## Data

The offline data can be found [here](https://drive.google.com/drive/folders/1i6V_t6TqaTpUdUFpOJT3y3KraJjak-sa?usp=sharing).

"data.zip" (~5 GB) contains everything needed for evalution. Please unzip it and put it into the MJOLNIR folder.

For training, please also download "train.zip" (~9 GB), and put all "Floorplan" folders into `./data/thor_v1_offline_data`


## Evaluation 

Note that DITA needs to specify a different agent_type in both training and testing.  

#### DITA
```bash

python main.py --eval \
    --test_or_val test \
    --episode_type TestValEpisode \
    --load_model pretrained_models/DITA.dat \
    --model DITA \
    --results_json dita.json \
    --gpu-ids 0 \
    --load_JG_model pretrained_models/JudgeModel.dat \
    --agent_type SupervisedNavigationAgent
```
Evaluating the DITA model result in auto-generations of action log files for visulization. 

#### Others

If you have trained other models ("SAVN" or "GCN" or "MJOLNIR_O"), please evaluate them using the following command.

```bash

python main.py --eval \
    --test_or_val test \
    --episode_type TestValEpisode \
    --load_model [model_name] \
    --model MJOLNIR_O \
    --results_json mjolnir_o.json \
    --gpu-ids 0 \
    --agent_type NavigationAgent
    --judge_model False
```

Other model options are "SAVN" or "GCN" or "MJOLNIR_O".


## Visualization

Note that our visualization only supports DITA model.

```bash
cd visualization
python visualization.py --actionList ../dita_vis.log
```


## Train

Note that DITA needs to specify a different agent_type in both training and testing.  


#### MJOLNIR_O

```bash
python main.py \
    --title mjolnir_train \
    --model MJOLNIR_O \
    --gpu-ids 0\
    --workers 8
    --vis False
    --save-model-dir trained_models
    --agent_type NavigationAgent
    
```


Other model options are "SAVN" or "GCN" or "MJOLNIR_O".

#### DITA

```bash
python main.py \
    --title DITA_training \
    --model DITA \
    --gpu-ids 0\
    --workers 8
    --vis False
    --save-model-dir trained_models
    --agent_type SupervisedNavigationAgent
```

