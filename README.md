Towards Pluralistic Value Alignment: Aggregating Value Systems Through ℓp-Regression
===================
This repository contains the implementation of all the algorithms and data of the experimental section of the paper
"Towards Pluralistic Value Alignment: Aggregating Value Systems Through ℓp-Regression"
presented by Roger Lera-Leri, Filippo Bistaffa, Marc Serramia, Maite Lopez-Sanchez, and Juan A. Rodríguez-Aguilar, 
in Proceedings of International Conference on Autonomous Agents and Multi-Agent Systems. 780-788

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
