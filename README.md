# acse-6-individual-assignment-acse-yw2619
This program is used to implement a parallel version of Conway’s Game of Life under MPI.
In order to realise parallel computing, this program uses domain decomposition and peer-to-peer communication.
## Usage
### Windows: Visual Studio 2017/2019
1.create a new project, add submit.cpp as the source file

2.build project

3.run the command in the terminal: 

mpiexec -n [processor number] [project name] [steps] [boundary] [height of domain] [width of domain] [per]

boundary = 0: fixed boundaries

boundary = 1: periodical boundaries

per: percentage of alive cells in the domain

ex: mpiexec -n 5 Project1 20 0 100 100 0.2
### Linux
mpicxx -o hello submit.cpp

mpirun -np [processor number] ./hello [steps] [boundary] [height of domain] [width of domain] [per]

ex: mpirun -np 2 ./hello 10 1 100 100 0.2
## Visualisation
In the submit.cpp, in order to get output files, users need to firstly comment the line:#ifndef DO_TIMING. After that, every process generates their own output file at each step. video.ipynb enables to generate all output files and presents a grapical output to illustrate the evolution of life in this domain.
