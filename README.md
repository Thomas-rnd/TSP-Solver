# Travelling Salesman Problem (TSP) Solver

![Result exemple with the Self-Organizing Maps](https://github.com/Thomas-rnd/TSP_solver/blob/main/gif/kohonen_pka379.gif)

This repository contains an implementation of several algorithms that can be
used to find sub-optimal solutions for the Traveling Salesman Problem. The
instances of the problems that the program supports are `.tsp` files, which is
a widespread format in this problem. All the source code can be found in the
`src` directory.

## Getting Started

The notebook will present how well an algorithm succeed to resolve the Travelling Salesman Problem (TSP).

### Algorithm implemented

- 2-opt inversion
- Nearest neighbor search
- Genetic algorithm
- Kohonen Self-Organizing Maps

### Running the app locally

Clone the git repository

```
git clone https://github.com/Thomas-rnd/TSP_solver
cd TSP_solver
```

Then create a virtual environment with conda then activate it. For more details go to [Managing environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#create-env-file-manually)

```
conda create -n <env_name> -c conda-forge dash jupyter matplotlib-inline numpy pandas pillow plotly python-kaleido python scipy 
conda activate <env_name>
```

Or

```
conda env create -f environment.yml
conda activate <env_name>
```

Run the app

Run [Jupyter](https://jupyter.org/) in whatever way works for you. The simplest would be to run `pip install jupyter && jupyter notebook`.
Then type the command `jupyter notebook` and the program will instantiate a local server at `localhost:8888` (or another specified port).

Now you’re in the Jupyter Notebook interface, open the notebook [test_TSP_solver.ipynb](http://localhost:8888/notebooks/test_TSP_solver.ipynb) 

## Built With

- [Pandas](https://pandas.pydata.org) - Data analysis and manipulation
- [Numpy](https://pandas.pydata.org) - Numerical computing with Python
- [Plotly Python](https://plot.ly/python/) - Used to create the interactive plots
