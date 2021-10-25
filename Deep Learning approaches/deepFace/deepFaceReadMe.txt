To run deepFace.py make sure the following are installed:
python >= 3.8.5
annoy >=1.17.0 (pip install annoy)
deepface >= 0.0.49 (pip install deepface)

deepFace.py takes the following parameters:
argv[0] = number of iterations
argc[1] = force pre-process
	where True --> regenerates the deepFaceData.pkl file
	      False --> does not regenerate the deepFaceData.pkl file