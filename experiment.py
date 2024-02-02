# from stock_gbm import Stock_GBM
from stock_gbm_copy import Stock_GBM
from ppo import PPO, ActorCritic, Transition
import jax
import jax.numpy as jnp
from typing import Sequence, NamedTuple, Any, Dict
import matplotlib.pyplot as plt

ppo_config = {
    "LR": 2.5e-4,
    "NUM_ENVS": 2,
    "NUM_STEPS": 100,
    "UPDATE_EPOCHS": 4,
    "NUM_MINIBATCHES": 2,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,
    "CLIP_EPS": 0.2,
    "ENT_COEF": 0.01,
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 0.5,
    "ACTIVATION": "tanh",
    "ANNEAL_LR": True,
}
    
config = {
    "NUM_ENVS": 2,
    "NUM_STEPS": 100,
    "OUTER_STEPS": 100,
    "DEBUG": True,
}

class Metric(NamedTuple):
    info: jnp.ndarray
    inner_step: jnp.ndarray
    outer_step: int

rng = jax.random.PRNGKey(0)  
# Initialise environment
env = Stock_GBM()
env_params=env.default_params
action_space = env.action_space().n
observation_shape = env.observation_space(env_params).shape

# Initialise agents
network = ActorCritic(action_dim=action_space, activation=ppo_config["ACTIVATION"])
agent1 = PPO(network=network, observation_shape=observation_shape, config=ppo_config)
agent2 = PPO(network=network, observation_shape=observation_shape, config=ppo_config)
agent1_train_state = agent1.initialise(rng)
agent2_train_state = agent2.initialise(rng)

# Vmap envionments 
env.batch_reset = jax.vmap(
    env.reset, in_axes=(0, None), out_axes=(0, 0)
)

env.batch_step = jax.vmap(
    env.step,
    in_axes=(0, 0, 0, None),
    out_axes=(0, 0, 0, 0, 0),
)

# Vmap agents
agent1.batch_policy = jax.jit(
    jax.vmap(agent1.policy, in_axes=(None, 0, 0), out_axes=(0, 0, 0))
)
agent2.batch_policy = jax.jit(
    jax.vmap(agent2.policy, in_axes=(None, 0, 0), out_axes=(0, 0, 0))
)

# Initialise env
reset_rng = jax.random.split(rng, config["NUM_ENVS"])
observations, env_state = env.batch_reset(reset_rng, env_params)

# Define rollout function for NUM_STEPS
def _inner_rollout(carry, inner_step):
    (rng, observations, agent1_train_state, agent2_train_state, env_state, env_params) = carry
    observation1, observation2 = observations
    rng, _rng = jax.random.split(rng)
    rng_step = jax.random.split(_rng, config["NUM_ENVS"])
    action1, value1, log_prob1 = agent1.batch_policy(agent1_train_state, observation1, rng_step)
    action2, value2, log_prob2 = agent2.batch_policy(agent2_train_state, observation2, rng_step)

    rng, _rng = jax.random.split(rng)
    rng_step = jax.random.split(_rng, config["NUM_ENVS"])

    observations, env_state, rewards, dones, info = env.batch_step(
        rng_step, env_state, (action1, action2), env_params
    )
    inner_timestep = jnp.full_like(dones[1], inner_step, dtype=jnp.int32)
    # inner_timestep = jnp.array(inner_step)

    traj1 = Transition(dones[0], action1, value1, rewards[0], log_prob1, observation1, info[0], inner_timestep)
    traj2 = Transition(dones[1], action2, value2, rewards[1], log_prob2, observation2, info[1], inner_timestep)

    return (rng, observations, agent1_train_state, agent2_train_state, env_state, env_params), (traj1,traj2)



def _outer_rollout(carry, outer_step):
    vals, trajectories = jax.lax.scan(
    _inner_rollout,
    carry,
    jnp.arange(config["NUM_STEPS"]),
    length=config["NUM_STEPS"],
    )
    metric1 = Metric(info = trajectories[0].info, inner_step = trajectories[0].timestep, outer_step = outer_step )
    metric2 = Metric(info = trajectories[1].info, inner_step = trajectories[1].timestep, outer_step = outer_step )

    if config.get("DEBUG"):
        def callback(metric):
            return_values = metric.info["average_price"][metric.info["all_done"]]
            timesteps = metric.inner_step[metric.info["all_done"]] * config["NUM_ENVS"]
            for t in range(len(timesteps)):
                print(f"outer step= {metric.outer_step}, inner step={timesteps[t]}, episodic return={return_values[t]}")
        jax.debug.callback(callback, metric1)

    (rng, observations, agent1_train_state, agent2_train_state, env_state, env_params) = vals
    rng, _rng = jax.random.split(rng)
    agent1_train_state = agent1.update(agent1_train_state, trajectories[0], _rng)
    agent2_train_state = agent2.update(agent2_train_state, trajectories[1], _rng)

    return (rng, observations, agent1_train_state, agent2_train_state, env_state, env_params), (metric1, metric2)



vals, metrics = jax.lax.scan(
    _outer_rollout,
    (
        rng,
        observations,
        agent1_train_state,
        agent2_train_state,
        env_state,
        env_params,
    ),
    jnp.arange(config["OUTER_STEPS"]),
    length=config["OUTER_STEPS"],
)



average_price = metrics[0].info["average_price"].reshape(-1, 2)
done = metrics[0].info["all_done"].reshape(-1, 2)

# print(average_price)
return_values =average_price[done]

# Plotting
plt.figure()  # Create a new figure for the second plot
plt.plot(return_values)
plt.xlabel('Time Step')
plt.ylabel('return_values')
plt.title('loss')
plt.show()
plt.savefig('loss_curve_a1.png')

# metric1, metric2 = metrics

# return_values = metric1.average_price[metric1.done]
# timesteps = metric1.inner_step[metric1.done] * config["NUM_ENVS"]

# print(return_values)
# print(timesteps)


# vals, trajectories = jax.lax.scan(
# _inner_rollout,
# carry,
# jnp.arange(config["NUM_STEPS"]),
# length=config["NUM_STEPS"],
# )
# # print(trajectories[0].info)
# print(trajectories[0].info["average_price"][trajectories[0].info["all_done"]])
# print(trajectories[0].timestep[trajectories[0].info["all_done"]] * config["NUM_ENVS"])


# vals, stack = jax.lax.scan(
#     _outer_rollout,
#     (
#         rng,
#         observations,
#         agent1_train_state,
#         agent2_train_state,
#         env_state,
#         env_params,
#     ),
#     None,
#     length=config["OUTER_STEPS"],
# )

# print(stack)


# # carry = (rng, observations, agent1_train_state, agent2_train_state, env_state, env_params)
# # vals, trajectories = jax.lax.scan(
# # _inner_rollout,
# # carry,
# # None,
# # length=config["NUM_STEPS"],
# # )


# # rng, _rng = jax.random.split(rng)
# # rng_update = jax.random.split(_rng, config["NUM_ENVS"])

# # agent1_train_state = agent1.update(agent1_train_state, trajectories[0], _rng)




# # agent1.batch_update = jax.vmap(agent1.update, in_axes=(None, None, 0), out_axes=None)
# # agent1_train_state = agent1.batch_update(agent1_train_state, trajectories[0], rng_update)










    




