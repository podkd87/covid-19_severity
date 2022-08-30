### Covid-19 Severity prediction by hierachical transformer model 

* This github explain the details of our article <br>
"In-hospital real-time prediction of COVID-19 severity regardless of disease phase using electronic health records"

* Data Extraction <br>
The dataset of this study is extracted from boramah medical center, South Korea.<br>

* 8-hour dataset <br>
The 8 hour dataset includes the following:<br>
Vital signs (blood pressure, heart rate, body temperature, respiratory rate, Saturation)

* 24-hour dataset <br>
The 24-hour datsset consists of the following:<br>
daily symptoms (binary value), <br>
daily checked laboratory test (continuous value) <br>
daily checked antiviral agents (binary value) <br>
daily checked steroid (continuous value)

* Baseline demographics dataset <br>
The patient baseline characteristics are as follows: <br>
Age, sex, smoking status, alcohol consumption status, <br>
initial vital sign,  <br>
weight, height,  <br>
underlying disease <br>
   (DM, tuberculosis, dyslipidemia, heart disease,  <br>
   brain disease, lung disease, liver disease,  <br>
   hollow viscus disease, other disease), <br>
Previous history of admission <br>

** File explanation
- Figure <br>
The figure ipynb file describes how model results were handled and plotted. <br>
<br>
- The model train.ipynb <br>
The model train file describes how the model was defined and loads the dataset based on 8-hour, 24-hour, baseline data.

