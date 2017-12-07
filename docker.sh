

sudo docker ps
sudo docker ps -a
sudo docker image ls
sudo docker image prune
sudo docker container prune

# launches the latest TensorFlow GPU binary image in a Docker container.
# In this Docker container, you can run TensorFlow programs in a Jupyter notebook:

sudo nvidia-docker run -it -p 8888:8888 gcr.io/tensorflow/tensorflow:latest-gpu

# On your local machine:
# sudo ssh -i awsKeys.pem -L 443:127.0.0.1:8888 ubuntu@ec2-54-147-126-214.compute-1.amazonaws.com
sudo ssh -i 443:127.0.0.1:8888 z@192.168.3.2

# # launches the latest TensorFlow GPU binary image in a Docker container
# # from which you can run TensorFlow programs in a shell:
#
# sudo nvidia-docker run -it gcr.io/tensorflow/tensorflow:latest-gpu bash

# Enter the container
sudo docker ps
sudo docker exec -it 333420737b94 /bin/bash

# sudo docker exec -it <container name> /bin/bash

apt-get update -y
apt-get upgrade -y

# install terminal tools
apt install -y git \
		wget \
	  ncdu \
    tmux \
		htop \
		zip \
		vim \
		openssl \
		libsm6 \
		libxrender1 \
		libfontconfig1

apt autoremove

# install keras opencv tqdm

pip install \
	  opencv-python\
	  keras\
	  tqdm\
		pydot\
		shutil

pip install -U scikit-image

# # config jupyter notebook
# mkdir ssl
# cd ssl
# openssl req -x509 -nodes -days 365 -newkey rsa:1024 -keyout "cert.key" -out "cert.pem" -batch
#
# jupyter notebook --generate-config
# vi ~/.jupyter/jupyter_notebook_config.py

# You need to insert the following lines of Python code (e.g. at the start of the file):
#
# c = get_config()  # get the config object
# c.NotebookApp.certfile = u'/home/ubuntu/ssl/cert.pem' # path to the certificate we generated
# c.NotebookApp.keyfile = u'/home/ubuntu/ssl/cert.key' # path to the certificate key we generated
# c.IPKernelApp.pylab = 'inline'  # in-line figure when using Matplotlib
# c.NotebookApp.ip = '*'  # serve the notebooks locally
# c.NotebookApp.open_browser = False  # do not open a browser window by default when using notebooks
# c.NotebookApp.password = 'sha1:b592a9cf2ec6:b99edb2fd3d0727e336185a0b0eab561aa533a43'  # this is the password hash that we generated earlier.

git config --global user.name "MinxZ"

git clone https://github.com/MinxZ/fashion_mnist.git
git clone https://github.com/MinxZ/userful_tools.git

cd Dog-Breed-Identification
bash main.sh

git pull
git add *
git commit -m "add something"
git push origin master
