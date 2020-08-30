# OCDC - An Obsessive-Compulsive Disorder Classification System (Mental Health master project 2020)

## Setup

We use a __python 3.7__ environment for this project.
We recommend to use a virtual environment for this project.
One popular tool for that sake is [Anaconda](https://www.anaconda.com/products/individual) and we would encourage you to use it. :)

Once you've downloaded anaconda make sure to create a virtual env with python 3.7 (using the anaconda navigator GUI this is really easy - ask if you need help).

After creating the environment, run it by clicking the play button in the anaconda navigator.
If you also want to use conda from the cmd line and are working on OSX (like me)  you need to add it to the path like this:

`export PATH=~/anaconda3/bin:$PATH`

To activate your conda environment from the cmd line run

`conda activate master-project` where master-project is your environment name.

Then cd into the repo and install the requirements by running `pip install -r requirements.txt`.

Also if you don't have a python IDE yet, we can recommend [PyCharm](https://www.jetbrains.com/de-de/pycharm/download/#section=mac) for which we get a free professional licence with our HPI email address.

In pycharm you also need to configure your conda environment to be the default env. Do so by clicking in the bottom right corner on your Python interpreter (e.g. Python 3.7) and then on "Add interpreter". 
There you should be able to select the previously created conda env.
Select it and add the interpreter.

Also make sure that pytest is your default test runner. 
Therefore go to your IDE preferences (PyCharm --> Preferences) and then go to Tools --> Python Integrated Tools --> Default Test Runner.

Before committing please run all the tests using this command:
`python -m pytest`

## Guidelines

Let's try to write clean and generally good quality code with comments for each method/class from the beginning on.

We have `pylint` installed, so please write tests if possible. 
Also if you want to create a new feature, please create a separate `feature/...`-branch for it branching off of the `dev`-branch.
If you feel insecure at any point with git branching/PR's or anything like that don't hesitate to ask. :)

## Folder structure
```
mp-mental-health-2020
--data: Here the recorded data is put, intoor_positioning data and IMU data seperated from each other.
----indoor_positioning: Here the indoor positioning data is put.
----phyphox: Here the IMU data is put - this folder can be explicitly configured in the configuration file.
--src: All the source code.
----classification: All relevant methods for classification.
----data_reading: Methods for reading the data from the data folder.
----evaluation: Methods for evaluating the results of classification.
----experiments: Methods to start a classification and data analysis run with a certain configuration.
------config_files: Includes a JSON file in which one can write a configuration which influences the execution of classification and data analysis runs.
------output_experiments: Here all the results of classification and data analysis are being safed.
----features: Methods relevant to feature engineering.
----indoor_positioning: Methods relevant to the preprocessing of indoor positioning data.
----output: Methods for saving outputs from the scripts, such as figures.
----preprocessing: Methods relevant to the preprocessing of all the data.
----visualization: Methods for visualizing data and results.
--tests: Tests for the code.
```


## Running the system

In the project folder executed the following command:
`$ python3  src/__main__.py`
