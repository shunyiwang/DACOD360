import os

start_dir = os.getcwd()

# mahimahi
os.system("sudo apt-get install python-software-properties")
os.system("sudo sysctl -w net.ipv4.ip_forward=1")
os.system("sudo add-apt-repository -y ppa:keithw/mahimahi")
os.system("sudo apt-get -y update")
os.system("sudo apt-get -y install mahimahi")

# apache server
os.system("sudo apt-get -y install apache2")

# selenium
os.system("wget 'https://pypi.python.org/packages/source/s/selenium/selenium-2.39.0.tar.gz'")
os.system("sudo apt-get -y install python-setuptools python-pip xvfb xserver-xephyr tightvncserver unzip")
os.system("tar xvzf selenium-2.39.0.tar.gz")
selenium_dir = start_dir + "/selenium-2.39.0"
os.chdir( selenium_dir )
os.system("sudo python setup.py install" )
os.system("sudo sh -c \"echo 'DBUS_SESSION_BUS_ADDRESS=/dev/null' > /etc/init.d/selenium\"")

# py virtual display
os.chdir( start_dir )
os.system("sudo pip install pyvirtualdisplay==0.2.1")
os.system("wget 'https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb' ")
os.system("sudo dpkg -i google-chrome-stable_current_amd64.deb")
os.system("sudo apt-get -f -y install")

# tensorflow
os.system("sudo apt-get -y install python-pip python-dev")
os.system("sudo pip install tensorflow==1.1.0")

# tflearn
os.system("sudo pip install tflearn==0.3.1")
os.system("sudo apt-get -y install python-h5py")
os.system("sudo apt-get -y install python-scipy")

# matplotlib
os.system("sudo apt-get -y install python-matplotlib")

