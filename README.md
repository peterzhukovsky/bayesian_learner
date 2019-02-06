# Bayesian learner
A set of scripts to simulate a trial-by-trial Bayesian learner and fit parameters to data using softmax and Binomial likelihood link.
For more details on the algorithm used, please refer to "Bayesian learner - reversal modelling.pdf".
The discrete implementation of the model inspired by Prof Jill O'Reilly's implementation (http://hannekedenouden.ruhosting.nl/RLtutorial/html/BayesModel1.html); theoretical explanations by Prof O'Reilly and Prof Hanneke den Ouden as well as Dynamic Belief Model papers by Yu and Cohen (2008) were very helpful in developing the models.

# 1. Bayesian_learner.m
Symbolic function impementation of a Bayesian learner that plots the expected q values, real q values and a posteriors from a few trials. Likelihood link is a Bernoulli distribution.

# 2. Bayesian_learner_Binomial.m
Symbolic function impementation of a Bayesian learner that plots the expected q values, real q values and a posteriors from a few trials. Likelihood link is a Binomial distribution; memory size is the number of trials the model can "remember" at any given trial t.

# 3. Bayesian_learner_trialwiseH.m
Symbolic function impementation of a Bayesian learner that plots the expected q values, real q values and a posteriors from a few trials. Likelihood link is a Bernoulli distribution; but H is another free parameter that can change on each trial.

# 4. Bayesian_learner_fit2data.m
Discrete implementation of a Bayesian learner with softmax that generates probability density functions for a given dataset, just like in the reversal_learning repository. Beta and kappa in the softmax are treated as discrete and negative log(PDF) is minimized to select the best fit parameters. BIC and pseudo-r2 values are also calculated.

Please note that the marginalised likelihood of the data (for any set of parameters) is not included in the algorithm.
