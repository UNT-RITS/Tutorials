# About Python 

Python is one of the most popular high-level language on Talon3. 
Talon3 has many versions of python that you can use.

A Python code/program/script is a collection of commands in a file designed to be executed in a particular sequence in order to perform a specific task. 

Ways of running Python code:

* offline on your computer: A guide for installing python can be found [here](https://www.programiz.com/python-programming/first-program). 

  Pros: it can be used without an internet connection.
  
  Cons: it can't be shared easily and you can't have acces from any device.

* online on browser: [Google Colaboratory](https://colab.research.google.com/notebooks/intro.ipynb) (“Colab”, for short) is an online platform that allows you to write and execute Python in your browser.

  Pros: you can easily share the code with your collegues and work on together simultaneously; you can run only parts of the code, not the entire project everytime.
  
  Cons: doesn't have much memory you can play with.

* on HPC:
  - from your terminal
  - from browser

To see the available python version on Talon3, run
```
module avail python
```
In order to use one of these python builds, you must "load" the module.
For example, if you want to use the 3.6.5 version of python, run
```
module load python/3.6.5
``` 
You MUST first load one of these modules to use python on Talon3. These python modules were specifically build for Talon3. Using the system's python (/usr/bin) will result is problems with your calculations



