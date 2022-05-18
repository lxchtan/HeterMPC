# HeterMPC: A Heterogeneous Graph Neural Network for Response Generation in Multi-Party Conversations
This repository contains the source code for the _ACL 2022_ paper [HeterMPC: A Heterogeneous Graph Neural Network for Response Generation in Multi-Party Conversations](https://arxiv.org/pdf/2203.08500.pdf). Jia-Chen Gu, Chao-Hong Tan, Chongyang Tao, Zhen-Hua Ling, Huang Hu, Xiubo Geng, Daxin Jiang. <br>
Hopefully, code will be released at the beginning of May. Thanks for your patience. <br>


## Introduction
Recently, various response generation models for two-party conversations have achieved impressive improvements, but less effort has been paid to multi-party conversations (MPCs) which are more practical and complicated. 
Compared with a two-party conversation where a dialogue context is a sequence of utterances, building a response generation model for MPCs is more challenging, since there exist complicated context structures and the generated responses heavily rely on both interlocutors (i.e., speaker and addressee) and history utterances. 
To address these challenges, we present HeterMPC, a heterogeneous graph-based neural network for response generation in MPCs which models the semantics of utterances and interlocutors simultaneously with two types of nodes in a graph. 
Besides, we also design six types of meta relations with node-edge-type-dependent parameters to characterize the heterogeneous interactions within the graph. 
Through multi-hop updating, HeterMPC can adequately utilize the structural knowledge of conversations for response generation. 
Experimental results on the Ubuntu Internet Relay Chat (IRC) channel benchmark show that HeterMPC outperforms various baseline models for response generation in MPCs.

<div align=center><img src="image/model_overview.png" width=80%></div>

<div align=center><img src="image/automated_evaluation.png" width=80%></div>

## Python environment

The requirements package is in `requirements.txt`.

If you are using nvidia's GPU and CUDA version supports 10.2, you can use the following code to create the desired virtual python environment:

```shell
conda create -n heterMPC python=3.8
conda activate heterMPC
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=10.2 -c pytorch
conda install dgl-cuda10.2=0.5.3 -c dglteam 
pip install -r requirements.txt
```

> If something goes wrong, you can also refer to the `heterMPC.yaml` file to get the package.

## Instruction

First, unpack data files:

```shell
cd data
tar -zxvf ubuntu_data.tar.gz
```

Please refer to the shell file under the `run_shell` folder.


## Update 
[20220518] Upload model source codes and generation results. Evaluation metrics will be updated later.

Please keep an eye on this repository if you are interested in our work.
Feel free to contact us ({gujc,chtan}@mail.ustc.edu.cn) or open issues.
