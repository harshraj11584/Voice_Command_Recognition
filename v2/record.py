import pyaudio
import pickle
import wave
import numpy as np
import soundfile as sf
from python_speech_features import mfcc



print("Initializing Environment Bias. \n\nEnter number of simulations: ")
num_bias_sim = int(input())

bias = np.zeros((1,5))
envbias=np.zeros((4043,))
for bias_num in range(num_bias_sim):
	CHUNK = 1024 
	FORMAT = pyaudio.paInt16 #paInt8
	CHANNELS = 1
	RATE = 8000 #sample rate
	RECORD_SECONDS = 2
	WAVE_OUTPUT_FILENAME = "testab.wav"
	print("\n\nRunning Bias Simulation ",(bias_num+1),"...")
	p = pyaudio.PyAudio()
	stream = p.open(format=FORMAT,channels=CHANNELS,rate=RATE,input=True,frames_per_buffer=CHUNK) #buffer
	print("* recording")
	frames = []
	for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
		data = stream.read(CHUNK)
		frames.append(data) # 2 bytes(16 bits) per channel
	print("* done recording")
	stream.stop_stream()
	stream.close()
	p.terminate()
	wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
	wf.setnchannels(CHANNELS)
	wf.setsampwidth(p.get_sample_size(FORMAT))
	wf.setframerate(RATE)
	wf.writeframes(b''.join(frames))
	wf.close()
	b =WAVE_OUTPUT_FILENAME
	data, samplerate = sf.read(b)
	data=np.nan_to_num(np.array(data))
	x= len(data)
	if x>=25000:
		data = data[:-10]
		x= len(data)
	p = 25000-x
	l =0
	tests = np.empty([200,8086])
	new_data = np.empty([25000,])
	y1 = np.empty([25000,]) 
	y = int(p/2);
	for i in range(0,y-1):
		new_data[i] = y1 [i]
	for i in range(y,25000-p+y-1):
		new_data[i] = data[i-y]
	for i in range(25000-y,24999):
		new_data[i] = y1[i]
	data1 = mfcc(new_data,samplerate)
	data1=np.nan_to_num(np.array(data1))
	data = data1
	data = np.resize(data1,(8086,))
	envbias = np.resize(data1,(4043,))
	#data=np.array(list(data)+list(envbias))
	nIn = 8086
	nOut = 5
	#x = data


	data = np.nan_to_num(np.array(data))
	predF1 = []
	predF= []
	m=[]
	# m1 = pickle.load(open('1_lda.sav', 'rb'))
	# predF1.append(m1.predict(np.nan_to_num(np.array([data]))))
	# m.append('lda')
	m2 = pickle.load(open('2_rf300.sav', 'rb'))
	predF1.append(m2.predict(np.nan_to_num(np.array([data]))))
	m.append('rf300')
	m3 = pickle.load(open('3_rf150.sav', 'rb'))
	predF1.append(m3.predict(np.nan_to_num(np.array([data]))))
	m.append('rf150')
	m4 = pickle.load(open('4_rf40.sav', 'rb'))
	predF1.append(m4.predict(np.nan_to_num(np.array([data]))))
	m.append('rf40')
	# m5 = pickle.load(open('5_lda.sav', 'rb'))
	# predF1.append(m5.predict(np.nan_to_num(np.array([data]))))
	# m.append('lda')
	m6 = pickle.load(open('6_mlp50.sav', 'rb'))
	predF1.append(m6.predict(np.nan_to_num(np.array([data]))))
	m.append('mlp50')
	m7 = pickle.load(open('7_mlp1000500.sav', 'rb'))
	predF1.append(m7.predict(np.nan_to_num(np.array([data]))))
	m.append('mlp1000500')
	m8 = pickle.load(open('8_rf350.sav', 'rb'))
	predF1.append(m8.predict(np.nan_to_num(np.array([data]))))
	m.append('rf350')
	m9 = pickle.load(open('9_mlp1000.sav', 'rb'))
	predF1.append(m9.predict(np.nan_to_num(np.array([data]))))
	m.append('mlp1000')
	m10 = pickle.load(open('10_mlp100.sav', 'rb'))
	predF1.append(m10.predict(np.nan_to_num(np.array([data]))))
	m.append('mlp100')
	m11 = pickle.load(open('11_rf200.sav', 'rb'))
	predF1.append(m11.predict(np.nan_to_num(np.array([data]))))
	m.append('rf200')
	m12 = pickle.load(open('12_rf250.sav', 'rb'))
	predF1.append(m12.predict(np.nan_to_num(np.array([data]))))
	m.append('rf250')

	# m1 = pickle.load(open('1_lda.sav', 'rb'))
	# predF.append(m1.predict_proba(np.nan_to_num(np.array([data]))))
	# m.append('lda')
	m2 = pickle.load(open('2_rf300.sav', 'rb'))
	predF.append(m2.predict_proba(np.nan_to_num(np.array([data]))))
	m.append('rf300')
	m3 = pickle.load(open('3_rf150.sav', 'rb'))
	predF.append(m3.predict_proba(np.nan_to_num(np.array([data]))))
	m.append('rf150')
	m4 = pickle.load(open('4_rf40.sav', 'rb'))
	predF.append(m4.predict_proba(np.nan_to_num(np.array([data]))))
	m.append('rf40')
	# m5 = pickle.load(open('5_lda.sav', 'rb'))
	# predF.append(m5.predict_proba(np.nan_to_num(np.array([data]))))
	# m.append('lda')
	# m6 = pickle.load(open('6_mlp50.sav', 'rb'))
	# predF.append(m6.predict_proba(np.nan_to_num(np.array([data]))))
	# m.append('mlp50')
	# m7 = pickle.load(open('7_mlp1000500.sav', 'rb'))
	# predF.append(m7.predict_proba(np.nan_to_num(np.array([data]))))
	# m.append('mlp1000500')
	m8 = pickle.load(open('8_rf350.sav', 'rb'))
	predF.append(m8.predict_proba(np.nan_to_num(np.array([data]))))
	m.append('rf350')
	# m9 = pickle.load(open('9_mlp1000.sav', 'rb'))
	# predF.append(m9.predict_proba(np.nan_to_num(np.array([data]))))
	# m.append('mlp1000')
	m10 = pickle.load(open('10_mlp100.sav', 'rb'))
	predF.append(m10.predict_proba(np.nan_to_num(np.array([data]))))
	m.append('mlp100')
	m11 = pickle.load(open('11_rf200.sav', 'rb'))
	predF.append(m11.predict_proba(np.nan_to_num(np.array([data]))))
	m.append('rf200')
	m12 = pickle.load(open('12_rf250.sav', 'rb'))
	predF.append(m12.predict_proba(np.nan_to_num(np.array([data]))))
	m.append('rf250')

	predF=np.array(predF)
	spredF=sum(predF)/7.0
	print(spredF.shape)
	print(bias.shape)
	bias = np.array(bias) + np.array(spredF)
	print("total bias = ", bias)
	#print(['back	','forward', 'left  ', 'right', 'stop'])
	print("Current bias = ", spredF)


#print(sum(sum(predF)[0]/12.0))

# bias = np.sum([ 
# [0.15390164, 0.08287993, 0.48778627, 0.21257808, 0.06285408],
# [0.16404863, 0.13878023, 0.16187798, 0.30834059, 0.22695257],
# [0.19731537, 0.11124153, 0.42112499, 0.20229804, 0.06802007],
# [0.16574558, 0.08047327, 0.36543774, 0.30001287, 0.08833055],
# [0.15919114, 0.07327519, 0.43273568, 0.25585109, 0.0789469 ],
# [0.15737329, 0.09066322, 0.40525909, 0.25157609, 0.0951283 ],
# [0.1739152 , 0.11669464, 0.32874215, 0.26413553, 0.11651247],
# [0.18822499, 0.02865282, 0.45313861, 0.24883297, 0.08115061],
# [0.15374209, 0.35922326, 0.1049512 , 0.21308333, 0.16900012],
# [0.18053402, 0.07433345, 0.49091664, 0.18777484, 0.06644105],
# [0.16521796, 0.11654437, 0.30131396, 0.31815448, 0.09876924],
# [0.19372247, 0.07655859, 0.4449629 , 0.20148986, 0.08326618]	], axis = 0)
print("\n\ndone")
bias = bias/num_bias_sim
print("Bias = ", bias)






























while(True):
	print("\n\nEnter Command? [y/n]")
	choice = str(input())
	if choice=='n' or choice == 'N':
		break


	CHUNK = 1024 
	FORMAT = pyaudio.paInt16 #paInt8
	CHANNELS = 1
	RATE = 8000 #sample rate
	RECORD_SECONDS = 2
	WAVE_OUTPUT_FILENAME = "testab.wav"

	p = pyaudio.PyAudio()
	stream = p.open(format=FORMAT,channels=CHANNELS,rate=RATE,input=True,frames_per_buffer=CHUNK) #buffer
	print("* recording")
	frames = []
	for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
		data = stream.read(CHUNK)
		frames.append(data) # 2 bytes(16 bits) per channel
	print("* done recording")
	stream.stop_stream()
	stream.close()
	p.terminate()
	wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
	wf.setnchannels(CHANNELS)
	wf.setsampwidth(p.get_sample_size(FORMAT))
	wf.setframerate(RATE)
	wf.writeframes(b''.join(frames))
	wf.close()
	b =WAVE_OUTPUT_FILENAME
	data, samplerate = sf.read(b)
	data=np.nan_to_num(np.array(data))
	x= len(data)
	if x>=25000:
		data = data[:-10]
		x= len(data)
	p = 25000-x
	l =0
	tests = np.empty([200,8086])
	new_data = np.empty([25000,])
	y1 = np.empty([25000,])	
	y = int(p/2);
	for i in range(0,y-1):
		new_data[i] = y1 [i]
	for i in range(y,25000-p+y-1):
		new_data[i] = data[i-y]
	for i in range(25000-y,24999):
		new_data[i] = y1[i]
	data1 = mfcc(new_data,samplerate)
	data1=np.nan_to_num(np.array(data1))
	data = data1
	data = np.resize(data1,(4043,))
	data=np.array(list(data)+list(envbias))
	np.resize(data,(8086,))
	nIn = 8086
	nOut = 5
	#x = data



	predF1 = []
	predF= []
	m=[]
	# m1 = pickle.load(open('1_lda.sav', 'rb'))
	# predF1.append(m1.predict(np.array([data])))
	# m.append('lda')
	m2 = pickle.load(open('2_rf300.sav', 'rb'))
	predF1.append(m2.predict(np.array([data])))
	m.append('rf300')
	m3 = pickle.load(open('3_rf150.sav', 'rb'))
	predF1.append(m3.predict(np.array([data])))
	m.append('rf150')
	m4 = pickle.load(open('4_rf40.sav', 'rb'))
	predF1.append(m4.predict(np.array([data])))
	m.append('rf40')
	# m5 = pickle.load(open('5_lda.sav', 'rb'))
	# predF1.append(m5.predict(np.array([data])))
	# m.append('lda')
	# m6 = pickle.load(open('6_mlp50.sav', 'rb'))
	# predF1.append(m6.predict(np.array([data])))
	# m.append('mlp50')
	# m7 = pickle.load(open('7_mlp1000500.sav', 'rb'))
	# predF1.append(m7.predict(np.array([data])))
	# m.append('mlp1000500')
	m8 = pickle.load(open('8_rf350.sav', 'rb'))
	predF1.append(m8.predict(np.array([data])))
	m.append('rf350')
	# m9 = pickle.load(open('9_mlp1000.sav', 'rb'))
	# predF1.append(m9.predict(np.array([data])))
	# m.append('mlp1000')
	m10 = pickle.load(open('10_mlp100.sav', 'rb'))
	predF1.append(m10.predict(np.array([data])))
	m.append('mlp100')
	m11 = pickle.load(open('11_rf200.sav', 'rb'))
	predF1.append(m11.predict(np.array([data])))
	m.append('rf200')
	m12 = pickle.load(open('12_rf250.sav', 'rb'))
	predF1.append(m12.predict(np.array([data])))
	m.append('rf250')

	# m1 = pickle.load(open('1_lda.sav', 'rb'))
	# predF.append(m1.predict_proba(np.array([data])))
	# m.append('lda')
	m2 = pickle.load(open('2_rf300.sav', 'rb'))
	predF.append(m2.predict_proba(np.array([data])))
	m.append('rf300')
	m3 = pickle.load(open('3_rf150.sav', 'rb'))
	predF.append(m3.predict_proba(np.array([data])))
	m.append('rf150')
	m4 = pickle.load(open('4_rf40.sav', 'rb'))
	predF.append(m4.predict_proba(np.array([data])))
	m.append('rf40')
	# m5 = pickle.load(open('5_lda.sav', 'rb'))
	# predF.append(m5.predict_proba(np.array([data])))
	# m.append('lda')
	m6 = pickle.load(open('6_mlp50.sav', 'rb'))
	predF.append(m6.predict_proba(np.array([data])))
	m.append('mlp50')
	m7 = pickle.load(open('7_mlp1000500.sav', 'rb'))
	predF.append(m7.predict_proba(np.array([data])))
	m.append('mlp1000500')
	m8 = pickle.load(open('8_rf350.sav', 'rb'))
	predF.append(m8.predict_proba(np.array([data])))
	m.append('rf350')
	m9 = pickle.load(open('9_mlp1000.sav', 'rb'))
	predF.append(m9.predict_proba(np.array([data])))
	m.append('mlp1000')
	m10 = pickle.load(open('10_mlp100.sav', 'rb'))
	predF.append(m10.predict_proba(np.array([data])))
	m.append('mlp100')
	m11 = pickle.load(open('11_rf200.sav', 'rb'))
	predF.append(m11.predict_proba(np.array([data])))
	m.append('rf200')
	m12 = pickle.load(open('12_rf250.sav', 'rb'))
	predF.append(m12.predict_proba(np.array([data])))
	m.append('rf250')

	predF=np.array(predF)
	spredF=sum(predF)/7.0
	print(spredF.shape)

	print(['back	','forward', 'left  ', 'right', 'stop'])
	print(predF)
	#print(sum(sum(predF)[0]/12.0))

	# bias = np.sum([ 
	# [0.15390164, 0.08287993, 0.48778627, 0.21257808, 0.06285408],
	# [0.16404863, 0.13878023, 0.16187798, 0.30834059, 0.22695257],
	# [0.19731537, 0.11124153, 0.42112499, 0.20229804, 0.06802007],
	# [0.16574558, 0.08047327, 0.36543774, 0.30001287, 0.08833055],
	# [0.15919114, 0.07327519, 0.43273568, 0.25585109, 0.0789469 ],
	# [0.15737329, 0.09066322, 0.40525909, 0.25157609, 0.0951283 ],
	# [0.1739152 , 0.11669464, 0.32874215, 0.26413553, 0.11651247],
	# [0.18822499, 0.02865282, 0.45313861, 0.24883297, 0.08115061],
	# [0.15374209, 0.35922326, 0.1049512 , 0.21308333, 0.16900012],
	# [0.18053402, 0.07433345, 0.49091664, 0.18777484, 0.06644105],
	# [0.16521796, 0.11654437, 0.30131396, 0.31815448, 0.09876924],
	# [0.19372247, 0.07655859, 0.4449629 , 0.20148986, 0.08326618]	], axis = 0)
	# bias=bias/12.0
	#print(bias)

	spredF = spredF-bias
	spredF[0,1]=spredF[0,1]-0.4
	print(spredF)

	ans = np.argmax(spredF[0])
	print("\nAnswer:")
	if(ans==0):
		print (ans,'back')
	if(ans==1):
		print (ans,'forward')
	if(ans==2):
		print (ans,'left')
	if(ans==3):
		print (ans,'right')
	if(ans==4):
		print (ans,'stop'	)		
	







