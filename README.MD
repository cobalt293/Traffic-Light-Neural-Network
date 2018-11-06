# Traffic Light Simulator

The code in this project simulates an four way traffic intersection and allows a neural network to train off of data produced.  In `main.py` the user can fail over the traffic light to run from the neural network.


## 1. Getting Setup

The project requires python3.5+ with tensorflow 1.7.  I highly recommend using the gpu version of tensorflow called  tensorflow-gpu.  It requires several additional pieces of software from NVIDIA and the installation procedure found [here](https://www.tensorflow.org/install/gpu).

Once you have finished the NVIDIA installation and have python 3.5+ installed you can enter the following in your terminal:

    python3 -m pip install numpy pandas tensorflow-gpu==1.7.0



## 2.  Training the model

To train the model you can run `train_failure_model.py` this will create a `training_log.csv` that will be used to train and the model will be saved `FailureModel/saved_model/model`

    python3 train_failure_model.py

## 3.  Running the Simulation

Now that you have created training data, and trained the model it's time to run the simulation.  To do this run the `main.py` file.

    python3 main.py

## 4.  Checking Results
Once `main.py` has been run you can graph the results of the simulation by running `primary_vs_failover.py`

    python3 primary_vs_failover.py