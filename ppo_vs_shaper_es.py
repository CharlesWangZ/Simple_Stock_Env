import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Dict
import distrax
import functools
from ppo_agent_no_reset import ScannedRNN, ActorCriticRNN, PPO
from double_stock_fixed_time import Stock_GBM_MULTI
import matplotlib.pyplot as plt
import evosax
from evosax import OpenES, PGPE, ParameterReshaper, FitnessShaper, NetworkMapper
from evosax.utils import ESLog
import json
import wandb
wandb.login()
wandb.init(
    project="shaper_vs_ppo",
)


NUM_ENV = 100
NUM_STEPS = 10
OUTER_STEPS = 1500
NUM_ITERS = 1000
POP_SIZE = 100

ppo_config_1 = {
    # "LR": 2.5e-4,
    "NUM_ENVS": NUM_ENV,
    "NUM_STEPS": NUM_STEPS,
    "UPDATE_EPOCHS": 4,
    "NUM_MINIBATCHES": 4,
    "GAMMA": 1,
    "GAE_LAMBDA": 1,
    "CLIP_EPS": 0.2,
    "ENT_COEF": 0.1,
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 0.5,
    "ACTIVATION": "tanh",
    "ANNEAL_LR": True,
    "HIDDEN_SIZE": 128,
    "NETWORK_SIZE": 128,
    "NUM_UPDATES": NUM_ITERS,
    "DEBUG": False,
}

ppo_config_2 = {
    "LR": 2e-4,
    "NUM_ENVS": NUM_ENV,
    "NUM_STEPS": NUM_STEPS,
    "UPDATE_EPOCHS": 4,
    "NUM_MINIBATCHES": 4,
    "GAMMA": 1,
    "GAE_LAMBDA": 1,
    "CLIP_EPS": 0.2,
    "ENT_COEF": 0.08,
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 0.5,
    "ACTIVATION": "tanh",
    "ANNEAL_LR": True,
    "HIDDEN_SIZE": 32,
    "NETWORK_SIZE": 32,
    "NUM_UPDATES": OUTER_STEPS,
    "DEBUG": False,
}

config = {
    "NUM_ENVS": NUM_ENV,
    "NUM_STEPS": NUM_STEPS,
    "OUTER_STEPS": OUTER_STEPS,
}


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarrayn
    info: jnp.ndarray

# rng = jax.random.PRNGKey(822)  
rng = jax.random.PRNGKey(87654)  
env = Stock_GBM_MULTI()
env_params=env.default_params
action_space = env.action_space().n
observation_shape = env.observation_space(env_params).shape

network_1 = ActorCriticRNN(action_dim=action_space, config=ppo_config_1)
agent1_init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], ppo_config_1["HIDDEN_SIZE"])
rng, _rng = jax.random.split(rng)
init_x = (
    jnp.zeros(
        (1, config["NUM_ENVS"], *observation_shape)
    ),
    jnp.zeros((1, config["NUM_ENVS"])),
)
network_params_1 = network_1.init(_rng, agent1_init_hstate, init_x)
param_reshaper = ParameterReshaper(network_params_1)
# strategy = OpenES(POP_SIZE, 
#                   param_reshaper.total_params, 
#                   opt_name="adam", 
#                   lrate_init=0.01,
#                 #   sigma_init=0.03,
#                 #   sigma_decay=0.999,
#                 #   sigma_limit=0.01,
#                 #   lrate_decay=0.999,
#                   )

# strategy = OpenES(POP_SIZE, 
#                   parmam_reshaper.total_params, 
#                   opt_name="adam", 
#                   lrate_init=0.008,
#                   sigma_init=0.03,
#                   sigma_decay=0.9999,
#                   sigma_limit=0.03,
#                   lrate_decay=0.9999,
#                   lrate_liit=0.001
#                   )


# In Figure
strategy = OpenES(POP_SIZE, 
                  param_reshaper.total_params, 
                  opt_name="adam", 
                  lrate_init=0.005,
                  sigma_init=0.03,
                  sigma_decay=0.98,
                  sigma_limit=0.04,
                  lrate_decay=0.99,
                #   lrate_limit=0.001
                  )

# strategy = OpenES(POP_SIZE, 
#                   param_reshaper.total_params, 
#                   opt_name="adam", 
#                   lrate_init=0.005,
#                   lrate_decay=0.99,
#                   sigma_init=0.04,
#                   sigma_decay=0.999,)
es_params = strategy.default_params


fit_shaper = FitnessShaper(w_decay=0.1,
                           maximize=True,)
rng, _rng = jax.random.split(rng)
state = strategy.initialize(_rng, es_params)

network_2 = ActorCriticRNN(action_dim=action_space, config=ppo_config_2)
agent2 = PPO(network=network_2, config=ppo_config_2)
agent2_train_state, agent2_init_hstate = agent2.initialise(observation_shape=observation_shape,rng=rng)
# print(agent2_init_hstate[0].shape[-1])

# Vmap envionments 
env.batch_reset = jax.vmap(
    env.reset, in_axes=(0, None), out_axes=(0, 0)
)

env.batch_step = jax.vmap(
    env.step,
    in_axes=(0, 0, 0, None),
    out_axes=(0, 0, 0, 0, 0),
)

# Initialise env
# reset_rng = jax.random.split(rng, config["NUM_ENVS"]) 
# observations, env_state = env.batch_reset(reset_rng, env_params)

# Define rollout function for NUM_STEPS
def _inner_rollout(carry, inner_step):
    (params, agent2_train_state, env_state, last_obsv, last_done, env_params, agent1_hstate, agent2_hstate, rng) = carry
    last_obs1, last_obs2 = last_obsv
    last_d1, last_d2 = last_done

    rng, _rng = jax.random.split(rng)
    ac_in = (last_obs1[np.newaxis, :], last_d1[np.newaxis, :])
    agent1_hstate, pi_1, value_1 = network_1.apply(params, agent1_hstate, ac_in)
    action_1 = pi_1.sample(seed=_rng)
    action_1 = jnp.where(env_state.time_left <= 0, env_state.quant_remaining[:,0], action_1)
    action_1 = jnp.clip(action_1, 0, env_state.quant_remaining[:,0])
    log_prob_1 = pi_1.log_prob(action_1)
    value_1, action_1, log_prob_1 = (
        value_1.squeeze(0),
        action_1.squeeze(0),
        log_prob_1.squeeze(0),
    )

    rng, _rng = jax.random.split(rng)
    agent2_hstate, pi_2, value_2 = agent2.policy(agent2_train_state, last_obs2, last_d2, agent2_hstate)
    action_2 = pi_2.sample(seed=_rng)
    action_2 = jnp.where(env_state.time_left <= 0, env_state.quant_remaining[:,1], action_2)
    action_2 = jnp.clip(action_2, 0, env_state.quant_remaining[:,1])
    log_prob_2 = pi_2.log_prob(action_2)
    value_2, action_2, log_prob_2 = (
        value_2.squeeze(0),
        action_2.squeeze(0),
        log_prob_2.squeeze(0),
    )

    # last_a2 = env_state.last_action[:,0]
    # action_1 = 0 * (action_2)
    # action_1 = jnp.where(env_state.time_left <= 0, env_state.quant_remaining[:,0], action_1)
    # action_1 = jnp.clip(action_1, 0, env_state.quant_remaining[:,0])

    rng, _rng = jax.random.split(rng)
    rng_step = jax.random.split(_rng, config["NUM_ENVS"])
    observations, env_state, rewards, dones, info = env.batch_step(
        rng_step, env_state, (action_1, action_2), env_params
    )
    traj1 = Transition(last_d1, action_1, value_1, rewards[0], log_prob_1, last_obs1, info[0])
    traj2 = Transition(last_d2, action_2, value_2, rewards[1], log_prob_2, last_obs2, info[1])
    carry = (params, agent2_train_state, env_state, observations, dones, env_params, agent1_hstate, agent2_hstate, rng)
    return carry, (traj1,traj2)

def _outer_rollout(carry, unused):
    agent2_init_hstate = carry[-2]
    carry, traj_batch = jax.lax.scan(
        _inner_rollout,
        carry,
        None,
        length=config["NUM_STEPS"]
    )
    params, agent2_train_state, env_state, observations, dones, env_params, agent1_hstate, agent2_hstate, rng = carry
    agent2_train_state, metric2 = agent2.update(traj_batch[1], observations[1], dones[1], agent2_init_hstate, agent2_hstate, agent2_train_state, rng)
    agent2_hstate = agent2.reset()
    carry = (params, agent2_train_state, env_state, observations, dones, env_params, agent1_hstate, agent2_hstate, rng)
    return carry, (*traj_batch, metric2)

def rollout(rng, params, agent1_init_hstate, agent2_train_state, agent2_init_hstate, env_params):
    reset_rng = jax.random.split(rng, config["NUM_ENVS"])
    observations, env_state = env.batch_reset(reset_rng, env_params)
    carry = (params, agent2_train_state, env_state, observations, (jnp.zeros((config["NUM_ENVS"]), dtype=bool), jnp.zeros((config["NUM_ENVS"]), dtype=bool)) , env_params, agent1_init_hstate, agent2_init_hstate, rng)
    carry, (traj_1, traj_2, metric2) = jax.lax.scan(
        _outer_rollout,
        carry,
        None,
        config["OUTER_STEPS"]
    )
    # params, agent2_train_state, env_state, observations, dones , env_params, agent1_hstate, agent2_hstate, rng = carry
    rewards_1 = traj_1.reward.mean()
    rewards_2 = traj_2.reward.mean()
    # action_1 = jnp.array([traj_1.action[0,:,0], traj_1.action[20,:,0], traj_1.action[50,:,0], traj_1.action[100,:,0], traj_1.action[200,:,0], traj_1.action[300,:,0], traj_1.action[400,:,0], traj_1.action[500,:,0], traj_1.action[600,:,0], traj_1.action[700,:,0], traj_1.action[800,:,0], traj_1.action[900,:,0], traj_1.action[1000,:,0]])
    # action_2 = jnp.array([traj_2.action[0,:,0], traj_2.action[20,:,0], traj_2.action[50,:,0], traj_2.action[100,:,0], traj_2.action[200,:,0], traj_2.action[300,:,0], traj_2.action[400,:,0], traj_2.action[500,:,0], traj_2.action[600,:,0], traj_2.action[700,:,0], traj_2.action[800,:,0], traj_2.action[900,:,0], traj_2.action[1000,:,0]])
    action_1 = jnp.array([traj_1.action[0,:,0], traj_1.action[20,:,0], traj_1.action[50,:,0], traj_1.action[100,:,0], traj_1.action[150,:,0], traj_1.action[200,:,0], traj_1.action[250,:,0], traj_1.action[300,:,0], traj_1.action[350,:,0], traj_1.action[400,:,0], traj_1.action[450,:,0], traj_1.action[500,:,0], traj_1.action[600,:,0], traj_1.action[700,:,0], traj_1.action[800,:,0], traj_1.action[900,:,0], traj_1.action[1000,:,0], traj_1.action[1200,:,0], traj_1.action[1500,:,0], traj_1.action[2000,:,0]])
    action_2 = jnp.array([traj_2.action[0,:,0], traj_2.action[20,:,0], traj_2.action[50,:,0], traj_2.action[100,:,0], traj_2.action[150,:,0], traj_2.action[200,:,0], traj_2.action[250,:,0], traj_2.action[300,:,0], traj_2.action[350,:,0], traj_2.action[400,:,0], traj_2.action[450,:,0], traj_2.action[500,:,0], traj_2.action[600,:,0], traj_2.action[700,:,0], traj_2.action[800,:,0], traj_2.action[900,:,0], traj_2.action[1000,:,0], traj_2.action[1200,:,0], traj_2.action[1500,:,0], traj_2.action[2000,:,0]])
    # action_2_traj = traj_2.action.reshape(-1, traj_2.action.shape[-1])
    # action_2 = action_2_traj[:10, 0], action_2_traj[200:210, 0], action_2_traj[500:510, 0], action_2_traj[1000:1010, 0], action_2_traj[2000:2010, 0], action_2_traj[3000:3010, 0], action_2_traj[4000:4010, 0], action_2_traj[5000:5010, 0], action_2_traj[6000:6010, 0], action_2_traj[7000:7010, 0], action_2_traj[8000:8010, 0], action_2_traj[9000:9010, 0], action_2_traj[10000:10010, 0], action_2_traj[12000:12010, 0], action_2_traj[15000:15010, 0] , action_2_traj[19990:20000, 0]

    return (rewards_1, rewards_2, action_1, action_2)
    # return (rewards_1, rewards_2, traj_1.action, traj_2.action)

# Fitness_top = []
# Fitness_ave = []
# Other_Fitness_top = []
# Other_Fitness_ave = []
for i in range(NUM_ITERS):
    rng, rng_init, rng_ask, rng_eval = jax.random.split(rng, 4)
    x, state = strategy.ask(rng_ask, state, es_params)
    params = param_reshaper.reshape(x)
    agent1_init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], ppo_config_1["HIDDEN_SIZE"])
    agent2_train_state, agent2_init_hstate = agent2.initialise(observation_shape=observation_shape,rng=rng)

    batch_rollout = jax.vmap(rollout, in_axes=(None, 0, None, None, None, None), out_axes=(0,0,0,0))
    # (fitness, other_fitness, traj_action_1, traj_action_2) = rollout(rng, network_params_1, agent1_init_hstate, agent2_train_state, agent2_init_hstate, env_params)

    (fitness, other_fitness, traj_action_1, traj_action_2) = batch_rollout(rng_eval, params, agent1_init_hstate, agent2_train_state, agent2_init_hstate, env_params)
    
    i_max = jnp.argmax(fitness)
    print(f"Iteration{i}")
    print("agent1_fitness:",fitness)
    print("agent2_fitness:",other_fitness)
    print("ACTION_1 after first, 20, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1500, 2000 updates")
    print(traj_action_1[i_max])
    print("ACTION_2 after first, 20, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1500, 2000 updates")
    print(traj_action_2[i_max])
    
    fitness_re = fit_shaper.apply(x, fitness)

    state = strategy.tell(x, fitness_re, state, es_params)
    # log = es_logging.update(log, x, fitness)
    # Fitness_top.append(fitness[i_max])
    # Fitness_ave.append(fitness.mean())
    # Other_Fitness_top.append(other_fitness[i_max])
    # Other_Fitness_ave.append(other_fitness.mean())
    print("Generation: ", i, "Top Performance: ", fitness[i_max], "Ave Performance", fitness.mean())
    wandb.log({"Fitness_top": fitness[i_max], "Fitness_ave": fitness.mean(), "Other_Fitness_top": other_fitness[i_max], "Other_Fitness_ave": other_fitness.mean(), "IS_1": -10.81, "IS_2": -11.81})
    # wandb.log({"IS_1": -10.81, "IS_2": -10.57,})
