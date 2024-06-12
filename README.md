# Trans-XFed

## Architecture
Code for the Paper "Trans-XFed: A Explainable Federated Learning for Supply
Chain Credit Assessment" 

[Trans-XFed]<img src='https://github.com/JieJieNiu/Trans-XFed/blob/main/image/transxfed.png' width='700'>



**>>>IMPORTANT<<<**

The original Code from the paper can be found in this branch:[[Tran-XFed](https://github.com/JieJieNiu/Trans-XFed)]


To choose the baseline model, go to args.py 


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
