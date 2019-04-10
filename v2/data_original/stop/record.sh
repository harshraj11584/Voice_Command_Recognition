x=1
i=22
while [ $x -le 5 ]
do
	read y
	if [ $y="/" ]
	then	
		echo Recording $i
		arecord back$i.wav
		((i++))
	fi

done
