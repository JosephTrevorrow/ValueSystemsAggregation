Hierarchical Consensus Value Aggregation for Value-Aligned Multi-Agent Systems
===================
This Repository contains all code and experiment data for the paper "Hierarchical Consensus Value Aggregation for Value-Aligned Multi-Agent Systems". This work is an extension of work done by researchers credited below, and as such a fork of a repository for their paper "Aggregating Value Systems for Decision Support". 

We run our experiments in "experiment_runner.py" which given the name of a society data will produce limit graph csv files for each timestep, and files contianing the consensus value system at each time step, as well as an average satisfaction for each group of agents. 

Usage testing on a society data file of "util_soc.csv: 
```
python-jl -m IPython experiment_runner.py util_soc
```

Aggregating Value Systems
===================
This repository contains the implementation of the algorithms and data of the experimental section of the paper
"Aggregating Value Systems for Decision Support" by Roger X. Lera-Leri, Enrico Liscio, Filippo Bistaffa, Catholijn M. Jonker, Maite Lopez-Sanchez, Pradeep K. Murukannaiah, Juan A. Rodríguez-Aguilar, and Francisco Salas-Molina
in Knowledge-Based Systems, 2024.

Dependencies
----------
 - [Python 3](https://www.python.org/downloads/)
 - [Julia](https://julialang.org/downloads/) and [PyJulia](https://pyjulia.readthedocs.io/en/latest/installation.html)
 - [Pandas](https://pandas.pydata.org/)
 - [Csv library](https://docs.python.org/3/library/csv.html)
 - [Numpy](https://numpy.org/)
 - [CVXPY](https://www.cvxpy.org/)
 - [ECOS](https://web.stanford.edu/~boyd/papers/ecos.html) (Embedded within CVXPY)

Dataset
----------
All experiments consider the European Values Study 2017: Integrated Dataset (EVS 2017) ([dataset](https://search.gesis.org/research_data/ZA7500?doi=10.4232/1.13560)).

Execution
----------
Our approach must be executed by means of the [`solve.py`](solve.py) Python script, i.e.,
```
usage: solve.py [-h] [-p P] [-e E] [-f F] [-w W] [-i I] [-o O] [-v] [-l] [-t]
                [-g G]

optional arguments:
  -h, --help  show this help message and exit
  -p P        p-norm (default: 2)
  -e E        epsilon used to compute limit p (default: 1e-4)
  -f F        CSV file with data (default: 'data.csv')
  -w W        weighting countries: 0 for unweighted problem, 1 for considering people that participated in the study and 2 for country population (default: 0)
  -i I        computes equivalent p given an input consensus
  -o O        write consensus to file
  -v          computes the preference aggregation
  -l          compute the limit p
  -t          compute the threshold p
  -g G        store results in csv
```


Acknowledgements
----------
This repository contains the [implementation of the pIRLS algorithm](https://github.com/fast-algos/pIRLS) ([article](https://papers.nips.cc/paper/2019/hash/46c7cb50b373877fb2f8d5c4517bb969-Abstract.html)). This article should be cited when citing our work.

Running the code
----------
- Install requirements: `pip install -r requirements.txt`
- To run locally with PyJulia compatability issues
  - Run `python-jl -m pip install IPython` to install IPython in Julia
  - Run your command as usual using `python-jl -m IPython` instead of `python`

As an example: `python-jl -m IPython  solve.py -f toy_data.csv -v true -g results.csv` wil:
- Load the toy_data.csv file
- Compute the preference aggregation
- Store the results in results.csv
