# Trans-XFed

## Architecture
Code for the Paper "Trans-XFed: A Explainable Federated Learning for Supply
Chain Credit Assessment" 

[Trans-XFed]<img src='https://github.com/JieJieNiu/Trans-XFed/blob/main/image/transxfed.png' width='700'>



**>>>IMPORTANT<<<**

The original Code from the paper can be found in this branch:[[Tran-XFed](https://github.com/JieJieNiu/Trans-XFed)]

The server_he.py is the central server with homomopic encryption.
The client.py is the local model training.

To choose the baseline model, prameters, clients' number, go to args.py 


The best trained model from 50 communication round canbe download in this branch,  and use it directly, also the baseline models of FedProx and FedAvg can be download here: 
[[Download trained models]( https://github.com/JieJieNiu/Trans-XFed/tree/main/model_save)]

The current master branch has since upgraded packages and was refactored. Since the exact package-versions differ the experiments may not be 100% reproducible.

If you have problems running the code, feel free to open an issue here on Github.

---

## Installing dependencies
In any case a [requirements.txt](requirements.txt) is also added from the poetry export.
```
pip install -r requirements.txt
```

Basically, only the following requirements are needed:
```
tenseal==0.3.14
numpy==1.20.3
pandas==1.3.4
captum==v0.7.0
torch==1.8.1
torchsummary==1.5.1
torchvision==0.9.1
```
---

## Usages
### Training
We offer several training options as below

For method (--M):
'TransFed': Trans-XFed
'FedProx': FedProX
'FedAvg':FedAvg

For loss Function (--loss):
'NLL': negative log-likelihood loss
'WCE': cross entropy loss
'Focal': Focal loss

For clients selection strategy:
'poc=True': performance-based clients selection
'poc=True': random clients selection

For proxmal term:
'fixmu=True': epoch-varying parameters
'fixmu=False': Determined parameters

For batchsize (--batchsize, default 64)
For training/testing epoch (--epoch, default 200)
Communication rounds: 50


example
For  training and testing:
```
python TransXFed\main.py
```


### Explanation
You can use [[[Integrated gradients](https://github.com/JieJieNiu/Trans-XFed/blob/main/intergrated_gradients1.py)] to interpret the results using saved best global model. 
Also, the visualization of IG and attention scores can be seen here: [Visulization](https://github.com/JieJieNiu/Trans-XFed/blob/main/plotresults.py)

### Plot
Ingegrated gradients:

defaulting
<img src='https://github.com/JieJieNiu/Trans-XFed/blob/main/image/ig_defaulting.png' width='700'>

non-defaulting
<img src='https://github.com/JieJieNiu/Trans-XFed/blob/main/image/ig_nondefaulting.png' width='700'>

Attention score:
defaulting<img src='https://github.com/JieJieNiu/Trans-XFed/blob/main/image/att_defaulting.jpg' width='200'> 
non-defaulting<img src='https://github.com/JieJieNiu/Trans-XFed/blob/main/image/att_nondefaulting.jpg' width='200'> 

### Dataset
The dataset are not open access due to the current data protocal.
If you are interested in the dataset that we used in this paper please write an e-mail to: s.mehrkanoon@uu.nl and j.shi1@uu.nl 

## Results
### Model performance

## Citation
If you decide to cite our project in your paper or use our data, please use the following bibtex reference:

```

---
