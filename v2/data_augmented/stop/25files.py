import numpy as np
import soundfile as sf
new_data = np.empty([25000,]) #creating an empty array for new file to be generated from original file
y1 = np.empty([25000,])
cnt=0
for j in range(1,121):
	print(j)
	b= "0stop"+str(j)+".wav"
	data, samplerate = sf.read(b) #reading audio file using soundfile library
	#print len(data), samplerate
	x= len(data)
	if x>=25000:
		data = data[:24900]
		x=len(data)
	p = 25000-x
	if (cnt>=2001):
		break
	for y in range(1 ,p,int(p/18)):  
		if (cnt>=2001):
			break 
		for i in range(0,y-1):      #adding empty elements in the array in the start
			new_data[i] =y1[i]
		for i in range(y,25000-p+y-1):
			new_data[i] =data[i-y]
		for i in range(25000-p+y , 24999):    #adding empty elements in the array in the end 
			new_data[i] = y1[i]	
		a = "stop"+str(cnt) +".wav"    #total length becomes 25000
		cnt= cnt+1
		sf.write(a, new_data, samplerate)  #audio files are written back to harddisk
		#print len(new_data)