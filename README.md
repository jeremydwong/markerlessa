# README #


### What is this repository for? ###

* 2024-2025 Git repository for analysing human reaching and walking data using markerlessmotion capture files from Google Mediapipe-backend programs (or similar). 

* Version 0.2

### Quick-start ###
For students: You may use either poetry or conda to install. 

Conda  
* Clone the repository
* in your terminal, move into that directory 
* ```conda env create -f environment.yml --arch arm64``` (macOS) or ```conda env create -f environment.yml --arch x86_64``` (windows)

Poetry
* Clone the repository
* in your terminal, move into the directory
* ```pyenv install 3.11.0 ```
* ```pyenv local 3.11.0 ``` # Creates .python-version file for this directory
* ```curl -sSL https://install.python-poetry.org | python3 - ```
* ```poetry env use $(pyenv which python)  ```# Points Poetry to pyenv's Python
* ``` poetry config installer.modern-installation false ``` # avoid problems identifying architecture. 
* ```poetry install ```


then,  
run the file src/go_test.py to test the installation via `python src/go_test.py`. 

### Contribution guidelines ###

* Writing code
it's recommended that we each create a python file for our own functions, and then import them into the main file. This way, we can work on our own functions without interfering with each other's work.

### Who do I talk to? ###

* Jeremy! Discord: _jdw
* Osman! Discord: _osmanda
