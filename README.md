# ReduceFA_2015
This private project is to detect life threatening arrhythmias in the intensive care unit.  It has similar that proposed in PhysioNet/CinC Challenge 2015, ["Reducing False Arrhythmia Alarms in the ICU"](https://www.physionet.org/challenge/2015/). 

PhysioNet challenge team described the severity of the problem as follow.
>False alarms in the ICU can lead to a disruption of care, impacting both the patient and the clinical staff through noise disturbances, desensitization to warnings and slowing of response times [1], leading to decreased quality of care [2,3]. ICU alarms produce sound intensities above 80 dB that can lead to sleep deprivation [1,4,5], inferior sleep structure [6,7], stress for both patients and staff [10,11,12,13] and depressed immune systems [14]. There are also indications that the incidence of re-hospitalization is lower if disruptive noise levels are decreased during a patient's stay [15]. Furthermore, such disruptions have been shown to have an important effect on recovery and length of stay [2,10]. In particular, cortisol levels have been shown to be elevated (reflecting increased stress) [12,13], and sleep disruption has been shown to lead to longer stays in the ICU [5]. ICU false alarm (FA) rates as high as 86% have been reported, with between 6% and 40% of ICU alarms having been shown to be true but clinically insignificant (requiring no immediate action) [16]. In fact, only 2% to 9% of alarms have been found to be important for patient management [17].

My contribution was to apply deep learning methodology to detect true arrhythmia signals from noisy ECG signals. At that time (and even now), deep learning was a brand new technology so there was only few researches that apply deep learing for ECG signal analysis. As far as I know, it was the first try to analyze ECG signal from patients in the ICU using deep learning, and experimental results show that the convolutional neural network yielded a sensitivity, specificity, and accuracy of 89.47%, 88.03%, and 88.67% in life-threatening arrhythmia detection. You can find more details in my master degree dissertation, **Detecting life threatening arrhythmias in the intensive care unit using deep learning algorithm**. (But sorry, I wrote it in Korean.)

## Codes
##### MATLAB code
This folder contains MATLAB code files that I used. 

**\ECGbeatclassification**
In this folder, there are code files that I tried to replicate Roshan Joy Martis' work,[ECG beat classification using PCA, LDA, ICA and Discrete Wavelet Transform](http://www.sciencedirect.com/science/article/pii/S1746809413000062) 
to check whether existing methods work well or not in my problem. Although my replication was not perfect (I could not get original codes that Roshan worked), It acheived similar results to the result in his paper.
However, I found that this approaches did not fit my problem well, because it is hard to find QRS peaks using general QRS detector such as Pan-Tompkins algorithm from ECG signal data offered from PhysioNet/CinC Challenge 2015.

**\ReduceFA_2015**
In this folder, there are code files that I used for data preprocessing and displaying results.

##### PhysioNetEasyDownload
This folder contains Python code files for collecting ECG signal data automatically from PhysioNet.

##### Theano_code
This folder contains [Theano](http://deeplearning.net/software/theano/) code files that I experimented. I used these codes for testing [deep belief network](http://deeplearning.net/tutorial/DBN.html).  

##### Torch_code
This folder contains [Torch7](http://torch.ch/) code files that I had experimented. This folder has my entire works, because I did most experiments using Torch. 

I had constructed my own deep learning architectures, such as deep neural networks, and deep convolutional networks. 
