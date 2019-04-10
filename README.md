# Voice_Command_Recognition  
Recognises Voice Commands recorded on System Microphone  

Dataset [[link](https://drive.google.com/open?id=1WphvzO0RzqXIQRkMgpSyIu6eJCS7rPst)] of 120 files each, for 5 Voice Commands [back,forward,left,right,stop] was recorded by [Lakshit](https://github.com/Lakshit-Singla)   
Each of those 120 files were augmented to 2002 files using **augmentor.py** , then put into **full_dataset** folder  

**Version 1** (v1) :  
All audio files converted to frequency domain from time domain using MFCC coefficients, then trained different Random Forest Classifiers and Simple MLPs with varying parameters.   
**Version 2** (v2) :  
Recorded Environment bias, trained on (recording,environment_bias) instead of just on recording. Got higher accuracy for same models as above.  

  Use **record.py** to test with laptop microphone. It first averages current environment bias, then lets you try voice commands.   




