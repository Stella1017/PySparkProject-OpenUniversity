# Term Project
## Data Analysis on Open University Learning Analytics Dataset

The completion rate of Massive Open Online Courses is a controversially topic and many researches are trying to understand the reasons behind the low completion rates. [1] This project leverages data from Open University Learning Analytics Dataset [2] to explore the factors that affect the completion rates and construct models to predict the probability of a student to pass or fail a course. This study focused on factors including demographic information of students and the course materials students interacted with. 
The accuracy of 2-class logistic regression is 82% and F1 score is 0.865. 

### Project Description
The project built three prediction models based on the demografic information of 1123 students who completed the BBB-2014B module and their interaction with course materials. I used the LogisticRegressionWithLBFGS and RandomForest.trainClassifier to build 2-class and 3-class prediction models.
There are 490 students withdrawn from the course. I also analyzed the trend of their withdrawn behaviors.

The FinalReport.pdf contains detailed information and analysis results.

## How to run  
To run the .py file, one needs to first download the files from https://analyse.kmi.open.ac.uk/open_dataset#data. One also needs to change the "path" in the code and run the codes line by line to get final results. 

```
Reference:
[1] Jordan, K. (2015). Massive open online course completion rates revisited: Assessment, length and attrition. International Review of Research in Open and Distributed Learning, 16(3) pp. 341–358.
[2] Kuzilek J., Hlosta M., Zdrahal Z. (2017) Open University Learning Analytics dataset Sci. Data 4:170171 doi: 10.1038/sdata.2017.171.
[3] Ahearn, A. (2017) The Flip Side of Abysmal MOOC Completion Rates? Discovering the Most Tenacious Learners.  https://www.edsurge.com/news/2017-02-22-the-flip-side-of-abysmal-mooc-completion-rates-discovering-the-most-tenacious-learners



