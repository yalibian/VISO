# VisML: Interactive Visualization for Machine Learning

## MOTIVATIONS
Our main idea is to explore how visualization with interactions [1] could improve the efficiency of communications between human and machine learning: how could interactive visualization could help users create machine learning algorithms effectively and efficiently.

  
## RELATED WORKS
Before our own design, we have explored the topic through both HCI and machine learning fields, and make a list of comparison of current research in HCI and Machine Learning. 
### 1. Machine Learning: 
The topic of effective machine learning creation has been explored in different perspectives in the machine learning fields: Active learning [2] tries to using less but the most valuable labels that provides the greatest information gain, evaluating effectiveness and efficiency as the numbers of examples users should label. Machine teaching [3] is an optimization problem over teaching examples that balance the future loss of the learner (machine learning algorithm) and the effort of the teacher. Interactive machine learning [4] tries to improve human teacher by giving teachings guidance, when human training an agent to perform a classification task by showing examples. The interactive machine learning also tries to improve the effectiveness of classifier creation through the efficiency of labeling. The guidance for users could improve the labeling efficiency based on provides users more information, the context about labeling could let users labeling quickly than labeling less examples without any context. 
Since current research on interactive machine learning are mainly works on how to provides users guides based on different presentation techniques based on ranking or neighbors. For our project, we plan to provides more context through visualization and let users labeling and explore dataset interactively. 

### 2. Visual Analytics:
Visual analytics [5] aims at providing visual platforms for users to interact directly with data and models through combining the power of automated data analysis and mining. Visual analytics researchers mainly explore the topic through how to make users effectively expressing their knowledge and preferences visually and interactively to data mining or machine learning methods. The visual classifier training model [6] presents an approach for building classifiers interactively and visually: how classifiers can be trained in a visual and interactive manner to improve recall and generalization for searching and filtering tasks in visual analytics scenarios. The Visual Classifier Training model mainly works on how to build classifiers interactively and visually in order to complement classical supervised machine learning techniques. 
Our project is also about creating a visual classifier training model, but providing more guidance based on interactive machine learning instead of active learning. We will also try to explore how different visualization layout could affect the labeling effect.

## PROGRESS
We got our idea of interactive machine learning powered with interactive visualization based on the combination of the visual classifier training model and Effective End-user Interaction with Machine Learning [7]. How could we interact with supervised machine learning in an observation level effectively - letting users explore big dataset and label items, sense-make current machine learning state in on visualization workspace. Instead of using several different visualization views to present the quality of the classifier, the status of the classifiers, the contents of the labels, we create only one visualization view (workspace) to let users explore the dataset and express their preference, and also let classifier re-layout the view to express the status, the quality of the classifiers. 
### 1. Dataset: 20 Newsgroups 
Since we are trying to develop a visual analytics application that combining both visualization with interactive machine learning. One of the most common used dataset for classification is the 20 Newsgroups. The 20 Newsgroups [8] data set is a collection of approximately 20,000 newsgroup documents in 20 different newsgroups. We first use calculated information gain to re-rank the features, and only take the first 5000 significant features as training.
### 2. Interactive Machine Learning
Based on the framework of interactive machine learning, the most basic interactive machine learning algorithms used in academic research is 'nearest-neighbor classifier' [9]: through re-rank samples with or without specific features, which could present examples illustrating its current understanding of the desired classifier and users use this presentation to decide how to further improve the system's understanding. Right now, we have use the basic KNN classification methods on Scikit-Learn to test current dataset. By setting the K value to 20, and rearrange the data matrix shape to fit the training function, we got a matching percentage around 26%, which is pretty low.
### 3. Interactive Visualization
For the basic visualization part, we will use the web-based version of StarSPIRE [10] that Yali Bian developed to visualize the text documents based on ForceSPIRE [10]. However, more interactions will be added to help users explore examples with context generated based on interactive machine learnings.  

## TIMELINE
Since we tried to combined works from both HCI and Machine Learning fields, we split the work into two parts: visualization, machine learnings. 
Time	Yali Bian	Liyan LI
10/18/17	Idea and related works	Idea and related works
11/1/17	Related Works and related visualization	Dataset to use on our idea
11/15/17	Basic Visualization design	Scikit-learn algorithms
11/28/17	Visualization Framework	KNN implementation on dataset
12/7/17	Merge Visualization to the framework	Merge Machine Learning to the frame work.
12/12/17	Case study	Evaluation 

## REFERENCES
1.	J.J. Thomas and K.A. Cook, eds., Illuminat- ing the Path: The Research and Development Agenda for Visual Analytics, Nat'l Visualization and Analytics Center, 2005.
2.	B. Settles, Active Learning, Morgan & Claypool, 2012.
3.	Simard P Y, Amershi S, Chickering D M, et al. Machine Teaching: A New Paradigm for Building Machine Learning Systems[J]. arXiv preprint arXiv:1707.06742, 2017.
4.	Fails, J.A., Olsen Jr., D.R. Interactive Machine Learning. IUI 2003, 39-45.
5.	J.J. Thomas and K.A. Cook, eds., Illuminat- ing the Path: The Research and Development Agenda for Visual Analytics, Nat'l Visualization and Analytics Center, 2005.
6.	Heimerl F, Koch S, Bosch H, et al. Visual classifier training for text document retrieval[J]. IEEE Transactions on Visualization and Computer Graphics, 2012, 18(12): 2839-2848.
7.	Amershi S. Designing for effective end-user interaction with machine learning[C]//Proceedings of the 24th annual ACM symposium adjunct on User interface software and technology. ACM, 2011: 47-50.
8.	http://qwone.com/~jason/20Newsgroups/
9.	Cover T, Hart P. Nearest neighbor pattern classification[J]. IEEE transactions on information theory, 1967, 13(1): 21-27.
10.	Bradel L, North C, House L. Multi-model semantic interaction for text analytics[C] Visual Analytics Science and Technology (VAST), 2014 IEEE Conference on. IEEE, 2014: 163-172. 
