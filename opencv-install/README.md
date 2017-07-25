install deb package using instructions from

	https://github.com/jabelone/OpenCV-for-Pi

    -------- Instructions ---------------------------------------------

    Always good practice to update everything before you install stuff:

       sudo apt-get update
       sudo apt-get upgrade
       sudo rpi-update

    We need to install some packages that allow OpenCV to process images:

       sudo apt-get install libtiff5-dev libjasper-dev libpng12-dev

    If you get an error about libjpeg-dev try installing this first:

       sudo apt-get install libjpeg-dev

    We need to install some packages that allow OpenCV to process videos:

       sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev

    We need to install the GTK library for some GUI stuff like viewing images.

       sudo apt-get install libgtk2.0-dev

    We need to install some other packages for various operations in OpenCV:

       sudo apt-get install libatlas-base-dev gfortran

    We need to install pip if you haven't done so in the past:

       wget https://bootstrap.pypa.io/get-pip.py
       sudo python get-pip.py

    Now we can install NumPy - a python library for maths stuff - needed for maths stuff.

       sudo pip install numpy

    Download and install the file from this repo called "latest-OpenCV.deb".

       wget "https://github.com/jabelone/OpenCV-for-Pi/raw/master/latest-OpenCV.deb"
       sudo dpkg -i latest-OpenCV.deb

    --------------------------------------------------------------------------------


make sim link from installed location to virtual enviroment

	first make a copy of the library with name cv2.so
                /usr/local/lib/python3.4/dist-packages
	then make simlink
		ln -s /usr/local/lib/python3.4/dist-packages/cv2.so cv2.so -> inside virtual env folder /home/pi/miniconda3/envs/pi/lib/python3.4/site-packages/cv2.so
	
