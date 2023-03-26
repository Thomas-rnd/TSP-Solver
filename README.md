# Travelling Salesman Problem (TSP) Solver

This repository contains an implementation of several algorithms that can be
used to find sub-optimal solutions for the Traveling Salesman Problem. The
instances of the problems that the program supports are =.tsp= files, which is
a widespread format in this problem. All the source code can be found in the
=src= directory.

## Getting Started

The notebook will present how well an algorithm succeed to resolve the Travelling Salesman Problem (TSP).

### Algorithm implemented

- 2-opt inversion
- Nearest neighbor search
- Genetic algorithm
- Kohonen Self-Organizing Maps

### Running the app locally

First create a virtual environment with conda or venv inside a temp folder, then activate it.

```
cd temp
python3 -m venv venv

# Windows
venv\Scripts\activate
# Or Linux and Mac
source venv/bin/activate
```

Clone the git repo, then install the requirements with pip

```
git clone https://github.com/Thomas-rnd/TSP_solver
cd dash_TSP
pip install -r requirements.txt
```

Run the app

Run [Jupyter](https://jupyter.org/) in whatever way works for you. The simplest would be to run `pip install jupyter && jupyter notebook`.

## Built With

- [Pandas](https://pandas.pydata.org) - Data analysis and manipulation
- [Numpy](https://pandas.pydata.org) - Numerical computing with Python
- [Plotly Python](https://plot.ly/python/) - Used to create the interactive plots
