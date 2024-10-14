# Simple_Stock_Env

single_stock simulates a Brownian Motion with votality(sigma) with one agent. Single_training trains a single PPO agent in the environment. Single PPO learns TWAP strategy through training.

ppo_vs_shaper_es is when shaper tries to shape the PPO agent into its favored strategies. Shaper achieves consistently better rewards than the PPO agent.

PPO[^2] class contains the functions to behave in the environment.

double_stock_fixed_time is a multi-agent training framework inspired by Pax: Scalable Opponent Shaping in JAX[^1]. Run ppo_vs_shaper_es for training.

[^1]: [PAX](https://github.com/ucl-dark/pax/tree/main).
[^2]: [Purejaxrl](https://github.com/luchris429/purejaxrl/tree/main)
