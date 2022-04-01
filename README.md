<H2> A. Particulars </H2>

<H3>Full Name: Karthik Prathaban<br>
Email Address: karthik93@outlook.sg

<H2> B. Folder Structure </H2>

```
├── src
│   ├── config.YAML          # Document for user to define pre-processing / machine learning parameters
│   ├── machine learning
│   │   ├── __init__.py
│   │   ├── models.py   
│   ├── preprocessing 
│   │   ├── __init__.py
│   │   ├── preprocess.py
│   ├── evaluation
│   │   ├── __init__.py
│   │   ├── model_eval.py  
├── data
│   ├── survive.db
├── eda.ipynb
├── README.md
├── requirements.txt
└── run.sh
```

<H2> C. Instructions for Executing Pipeline </H2>

1. Ensure python3 is installed: ```$ sudo apt-get install python3```
2. Navigate to the source folder (e.g. ```$ cd /mnt/c/exampleuser/documents/AIAP_Improved_Karthik_Prathaban``` in WSL)
3. Install required packages from the provided requirements folder: ```$ pip install -r requirements.txt```
4. *Optional*: Open **config.yaml** in the **/src** subfolder and alter the processing, machine learning and testing parameters per your preference 
5. Ensure run.sh is executable: ```chmod u+x run.sh```
6. Run the program ```./run.sh```

<H2> D. Steps / Flow of Pipeline </H2>

<img src = "Figures/Workflow.png" width="496" height="619">



<H2> E. EDA Findings </H2>
<H3> Data distributions and errors</H3>

1. Mssing values were observed for 'Creatinine'
2. Negative values were observed for 'Age'
3. Data 

<H3> 2. </H3>

<H2> F. Choice of ML Models </H2>

<H2> G. Model Evaluation </H2>

<H2> H. Other Considerations </H2>

<H2> Update </H2>