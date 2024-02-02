# Simple_Stock_Env

Stock_Env simulates a Brownian Motion with votality(sigma) with more then one agents. Each agent's action would have a price impact on the stock price. 

PPO class contains the functions to behave in the environment.

experiment.py is a multi-agent training framework inspired by Pax: Scalable Opponent Shaping in JAX[^1]. Run experiment.py for training.

[^1]: [supporting link](https://github.com/ucl-dark/pax/tree/main).
