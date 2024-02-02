import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
import time
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Dict
from flax.training.train_state import TrainState
import distrax
import gymnax
import functools
from gymnax.environments import spaces
import chex
from stock_gbm import Stock_GBM

class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)
     
class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    timestep: jnp.ndarray

class PPO():
    def __init__(
            self,
            network,
            observation_shape,
            config,
                 ):
        
        config["NUM_UPDATES"] = (
            config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
        )
        config["MINIBATCH_SIZE"] = (
            config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
        )
        
        def _policy(
                train_state: TrainState, observation: jnp.ndarray, key: chex.PRNGKey
        ):
            key, subkey = jax.random.split(key)
            dist, value = network.apply(train_state.params, observation)
            action, log_prob = dist.sample_and_log_prob(seed=subkey)
            return action, value, log_prob
        self.policy = _policy
        
        def _initialise(
                key: chex.PRNGKey
        ):
            rng, _rng = jax.random.split(key)
            init_x = jnp.zeros(observation_shape)
            network_params = network.init(_rng, init_x)
            def linear_schedule(count):
                    frac = (
                        1.0
                        - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
                        / config["NUM_UPDATES"]
                    )
                    return config["LR"] * frac
            if config["ANNEAL_LR"]:
                tx = optax.chain(
                    optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                    optax.adam(learning_rate=linear_schedule, eps=1e-5),
                )
            else:
                tx = optax.chain(
                    optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                    optax.adam(config["LR"], eps=1e-5),
                )

            train_state = TrainState.create(
                apply_fn=network.apply,
                params=network_params,
                tx=tx,
            )
            return train_state
        self.initialise = _initialise

        def gae_advantages(
                rewards: jnp.ndarray, values: jnp.ndarray, dones: jnp.ndarray
        ) -> jnp.ndarray:
            
            discounts = config["GAMMA"] * jnp.logical_not(dones)
            
            reverse_batch = (
                jnp.flip(values, axis=0),
                jnp.flip(rewards, axis=0),
                jnp.flip(discounts, axis=0),
            )

            def get_advantages(carry, transition):
                gae, next_value, gae_lambda = carry
                value, reward, discounts = transition
                value_diff = discounts * next_value - value
                delta = reward + value_diff
                gae = delta + discounts * gae_lambda * gae
                return (gae, value, gae_lambda), gae

            _, advantages = jax.lax.scan(
                get_advantages,
                (
                    jnp.zeros_like(values[-1]),
                    values[-1],
                    jnp.ones_like(values[-1]) * config["GAE_LAMBDA"],
                ),
                reverse_batch,
            )

            advantages = jnp.flip(advantages, axis=0)
            target_values = values + advantages  # Q-value estimates
            target_values = jax.lax.stop_gradient(target_values)
            return advantages, target_values

        def loss(
            params,
            traj_batch,
            advantages: jnp.array,
            target_values: jnp.array,
        ):
            # print("PARAM:",train_state.params)
            dists, values = network.apply(params, traj_batch.obs)
            log_prob = dists.log_prob(traj_batch.action)

            # Value loss
            value_pred_clipped = traj_batch.value + ( values - traj_batch.value ).clip(-config["CLIP_EPS"],config["CLIP_EPS"])
            value_losses = jnp.square(values - target_values)
            value_losses_clipped = jnp.square( value_pred_clipped - target_values )
            value_loss = ( 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean() )

            # Actor loss

            ratio = jnp.exp(log_prob - traj_batch.log_prob)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            loss_actor1 = ratio * advantages
            loss_actor2 = (
                jnp.clip(
                    ratio,
                    1.0 - config["CLIP_EPS"],
                    1.0 + config["CLIP_EPS"],
                )
                * advantages
            )
            loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
            loss_actor = loss_actor.mean()
            entropy = dists.entropy().mean()

            total_loss = (loss_actor + config["VF_COEF"] * value_loss - config["ENT_COEF"] * entropy)

            return total_loss, (value_loss, loss_actor, entropy)

        def _update(train_state, traj_batch, rng):
            def update_epoch(update_state, unused):
                train_state, traj_batch, rng = update_state
                advantages, targets = gae_advantages(traj_batch.reward, traj_batch.value, traj_batch.done)
                rng, _rng = jax.random.split(rng)
                # Batching and Shuffling
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                # Mini-batch Updates
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )

                def update_minibatch(
                        train_state: TrainState, batch_info
                ):
                    traj_batch, advantages, targets = batch_info
                    grad_fn = jax.value_and_grad(loss, has_aux = True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                # train_state, total_loss = jax.lax.scan(
                #     update_minibatch, 
                #     (train_state, minibatches),
                #     None
                # )

                train_state, total_loss = jax.lax.scan(
                    update_minibatch,
                    train_state,
                    minibatches,
                    length=config["NUM_MINIBATCHES"]
                )

                update_state = (train_state, traj_batch, rng)
                return update_state, total_loss
            
            update_state = train_state, traj_batch, rng

            update_state, loss_info = jax.lax.scan(
                update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            (train_state, traj_batch, rng) = update_state
            
            return train_state
        self.update = _update



if __name__ == "__main__":
    env = Stock_GBM()
    env_params=env.default_params
    rng = jax.random.PRNGKey(0)  

    config = {
            "LR": 2.5e-4,
            "NUM_ENVS": 2,
            "NUM_STEPS": 128,
            "TOTAL_TIMESTEPS": 5e5,
            "UPDATE_EPOCHS": 4,
            "NUM_MINIBATCHES": 4,
            "GAMMA": 0.99,
            "GAE_LAMBDA": 0.95,
            "CLIP_EPS": 0.2,
            "ENT_COEF": 0.01,
            "VF_COEF": 0.5,
            "MAX_GRAD_NORM": 0.5,
            "ACTIVATION": "tanh",
            "ENV_NAME": "CartPole-v1",
            "ANNEAL_LR": True,
            "DEBUG": True,
        }
    
    action_space = env.action_space().n
    observation_shape = env.observation_space(env_params).shape

    network = ActorCritic(action_space, activation=config["ACTIVATION"])
    agent1 = PPO(network=network, observation_shape=observation_shape, config=config)
    agent2 = PPO(network=network, observation_shape=observation_shape, config=config)

    agent1_train_state = agent1.initialise(rng)
    agent2_train_state = agent2.initialise(rng)

    reset_rng = jax.random.split(rng, config["NUM_ENVS"])
    policy_rng = jax.random.split(rng, config["NUM_ENVS"])

    env.reset = jax.vmap(
        env.reset, in_axes=(0, None), out_axes=(0, 0)
    )

    env.step = jax.vmap(
        env.step,
        in_axes=(0, 0, 0, None),
        out_axes=(0, 0, 0, 0, 0),
    )

    observations, env_state = env.reset(reset_rng, env_params)
    observation1, observation2 = observations
    print(observations)

    agent1.batch_policy = jax.jit(
        jax.vmap(agent1.policy, in_axes=(None, 0, 0), out_axes=(0, 0, 0))
    )
    agent2.batch_policy = jax.jit(
        jax.vmap(agent2.policy, in_axes=(None, 0, 0), out_axes=(0, 0, 0))
    )

    def _inner_rollout(carry, unused):
        (rngs, observations, agent1_train_state, agent2_train_state, env_state, env_params) = carry
        observation1, observation2 = observations
        action1, value1, log_prob1 = agent1.batch_policy(agent1_train_state, observation1, rngs)
        action2, value2, log_prob2 = agent2.batch_policy(agent2_train_state, observation2, rngs)

        observations, env_state, rewards, dones, info = env.step(
            rngs, env_state, (action1, action2), env_params
        )
        info["timestep"]
        traj1 = Transition(dones[0], action1, value1, rewards[0], log_prob1, observation1, info[0])
        traj2 = Transition(dones[1], action2, value2, rewards[1], log_prob2, observation2, info[1])

        return (rngs, observations, agent1_train_state, agent2_train_state, env_state, env_params), (traj1,traj2)

    rngs = jax.random.split(rng, config["NUM_ENVS"])
    carry = (rngs, observations, agent1_train_state, agent2_train_state, env_state, env_params)
    vals, trajectories = jax.lax.scan(
                    _inner_rollout,
                    carry,
                    None,
                    length=10,
                )

    # agent1.batch_update = jax.jit(jax.vmap(agent1.update, (None, 0, 0)))
    # agent1.batch_update(agent1_train_state,trajectories[0],rngs)

        










                
            

            

            
        



                
            










                
                
                
                
                











                
                
            


        
    # act_space = 
    # obs_space = 
    # network = ActorCriticRNN(act_space, config)
    # agent = PPO(network, )

    # env = 

    # action = policy(env.obs)
    # jax.lax.scan

    # agent.update(actions, obs, rewards, dones)




            
