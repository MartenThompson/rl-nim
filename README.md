# rl-nim - Reinforcement Learning on Nim

## Marten Thompson

Explores Reinforcement Learning on the strategy game Nim. 

Contents available on: <a href='https://github.com/MartenThompson/rl-nim.git'> GitHub </a>


### Contents
* `final_report.pdf` primary artifact and final report
* `code` directory
	* `agents.py` class definitions for Q and Bayes agents
	* `learning.py` primary script for generating results. Writes output to `final` directory.
	* `nimUtils.py`, `play.py`, `statesActions.py` all contain helper functions for the primary script `learning.py`.
	* `visualization.ipynb` reads from `final` directory to create the figures in the report. It also produces many more not used in the final report.
* `final` directory
	* `Bayes` directory: logs from Bayes training, shown in figure 6.c
	* `Bayes_vis` directory: logs to produce figure 4
	* `PvP` directory: logs from final best-vs-all simulations, as depicted by figure 5
	* `Q_vis` directory: logs to produce figures 2 and 3
	* `QtvQt` directory: logs from $Q_t$ training, a small number of which are shown in figure 6.b
	* `QvQ` directory: logs from Q training, a small subset of which results in figure 6.a



### Execution

This project did not require a special runtime environment. It used the latest versions of standard packages (`numpy`, `pandas`, `matplotlib`, `functools`, `operator`, `sys`, `csv`, `datetime`).

If you wish to recreate any of the results contained in the final report, execute `learning.py`. Within `main`, comment or uncomment calls to the following functions to reproduce results:

* `vis_learning()` (15 minutes): visualize changes in Q-table, figures 2 and 3. Note: this will overwrite the content in `final/Q_vis/`
* `QvQgrid()` (5 hours): the 75 trials between Q-agents detailed in report. This will add timestamped files to `final/QvQ/`
* `QtvQtgrid()` (20 minutes): trails between $Q_t$-agents. This will add timestamped files to `final/QtvQt/`
* `bayesAgents()` (2.5 hours): trails between Bayes agents. This will add timestamped files to `final/Bayes/`
* `bayesVis()` (20 minutes): visualize Bayesian prior moments. This will overwrite the contents in `final/Bayes_vis/`
* `BestvEachOther()` (1.5 hours): final comparison between the best agents. This will add timestamped files to `final/PvP/`
    
    
    
    
    
    
    
    
    
    
    