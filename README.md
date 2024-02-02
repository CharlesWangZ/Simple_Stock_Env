# Simple_Stock_Env

Stock_Env simulates a Brownian Motion with votality(sigma) with more then one agents. Each agent's action would have a price impact on the stock price. 

PPO[^2] class contains the functions to behave in the environment.

experiment.py is a multi-agent training framework inspired by Pax: Scalable Opponent Shaping in JAX[^1]. Run experiment.py for training.

[^1]: [PAX](https://github.com/ucl-dark/pax/tree/main).
[^2]: [Purejaxrl]https://github.com/luchris429/purejaxrl/tree/main
