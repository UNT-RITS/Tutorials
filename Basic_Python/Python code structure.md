# Python code structure

Content:
 - [Input](https://github.com/UNT-RITS/Tutorials/blob/master/Basic_Python/Python%20code%20structure.md#input)
 - [Output](https://github.com/UNT-RITS/Tutorials/blob/master/Basic_Python/Python%20code%20structure.md#output)
 - [Imports](https://github.com/UNT-RITS/Tutorials/blob/master/Basic_Python/Python%20code%20structure.md#import)
  
Here is a basic python program that plots the graph of the function f: R → R , where  f(x)= √x

![Example of a basic py code](https://github.com/UNT-RITS/Tutorials/blob/master/Basic_Python/images/code_structure.png)

As shown in the example above, the file can contain _imports, defined functions, built-in functions_, and so on. Almost any code you'll write will have an input, output and imports, along with the main commands and functions.

In order to become a python user you need to be aware of the integrated tools available for you to apply in your 
personal/school/work projects (Data-Analysis, Data processing, etc. …). 
 
## Input

Until now, the value of variables was defined. To allow flexibility in the program, sometimes we might want to take the input from the user. In Python, the input() function allows this. 
The syntax for input() is:

![](https://github.com/UNT-RITS/Tutorials/blob/master/Basic_Python/images/code_structure2.png)

Where ‘Your name is: ‘ can be replaced with what you need from the user. For example:

``` python 
input ("Insert a number: ")
```

We can see that the entered value, 5, is taken by the program as a string. 
It is important to know what kind of input you are expecting from the user. 
If you need a string – the above method works, but if you need an integer or a float to proceed with further calculations, you have to encapsulate the input() into int(input( … )) or float(input( … )). 
Bonus you can directly calculate a string operation using eval() on the input() as in the example below: 

![](https://github.com/UNT-RITS/Tutorials/blob/master/Basic_Python/images/code_structure3.png)

## Output
The _print()_ function is used to output data to the screen. We can also output [data to a file](https://www.programiz.com/python-programming/file-operation) (useful when run the code on HPC). 

``` python 
print ("My first print using Py")
print ('My send print')
```

This example shows that you can use both “ ” and ‘ ’ to print a string.
The following example shows how to print a description of a variable together with that variable.

``` python 
x = 2
print ('The value of the variable is:', x)
```

**Output formatting.** If you want a more rigorous output you can do this by using _str.format()_ method.
![](https://github.com/UNT-RITS/Tutorials/blob/master/Basic_Python/images/code_structure1.png)

## Import
