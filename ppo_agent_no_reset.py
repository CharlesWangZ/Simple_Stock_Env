import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from typing import Sequence, NamedTuple, Any, Dict
import distrax
import functools
from double_stock import Stock_GBM_MULTI
from flax.training.train_state import TrainState

# network structure
class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        features = carry[0].shape[-1]
        new_rnn_state, y = nn.GRUCell(features)(rnn_state, ins)
        return new_rnn_state, y
    
    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        return nn.GRUCell(hidden_size, parent=None).initialize_carry(
            jax.random.PRNGKey(0), (batch_size, hidden_size)
        )


class ActorCriticRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        embedding = nn.Dense(
            self.config["NETWORK_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = nn.Dense(self.config["NETWORK_SIZE"], kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(self.config["NETWORK_SIZE"], kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return hidden, pi, jnp.squeeze(critic, axis=-1)
    

# Defines major function for PPO agent, including action, policy, update and reset
class PPO():
    def __init__(self, network, config):
        def _initialise(observation_shape, rng):
            init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], config["HIDDEN_SIZE"])
            rng, _rng = jax.random.split(rng)
            init_x = (
                jnp.zeros(
                    (1, config["NUM_ENVS"], *observation_shape)
                ),
                jnp.zeros((1, config["NUM_ENVS"])),
            )
            network_params = network.init(_rng, init_hstate, init_x)
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
            return train_state, init_hstate
        self.initialise = _initialise

        def _policy(train_state, last_obs, last_done, hidden_state):
            ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
            hidden_state, pi, value = network.apply(train_state.params, hidden_state, ac_in)
            return hidden_state, pi, value
        self.policy = _policy

        def _calculate_gae(traj_batch, last_val, last_done):
            def _get_advantages(carry, transition):
                gae, next_value, next_done = carry
                done, value, reward = transition.done, transition.value, transition.reward 
                delta = reward + config["GAMMA"] * next_value * (1 - next_done) - value
                gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - next_done) * gae
                return (gae, value, done), gae
            _, advantages = jax.lax.scan(_get_advantages, (jnp.zeros_like(last_val), last_val, last_done), traj_batch, reverse=True, unroll=16)
            return advantages, advantages + traj_batch.value
        
        def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                        # RERUN NETWORK
                        _, pi, value = network.apply(
                            params, init_hstate[0], (traj_batch.obs, traj_batch.done)
                        )
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, init_hstate, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state

                rng, _rng = jax.random.split(rng)
                permutation = jax.random.permutation(_rng, config["NUM_ENVS"])
                batch = (init_hstate, traj_batch, advantages, targets)

                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )

                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], config["NUM_MINIBATCHES"], -1]
                            + list(x.shape[2:]),
                        ),
                        1,
                        0,
                    ),
                    shuffled_batch,
                )

                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, total_loss

        def _update(traj_batch, last_obs, last_done, init_hstate ,hstate, train_state, rng):
            ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
            _, _, last_val = network.apply(train_state.params, hstate, ac_in)
            last_val = last_val.squeeze(0)
            advantages, targets = _calculate_gae(traj_batch, last_val, last_done)

            init_hstate = init_hstate[None, :]
            update_state = (train_state, init_hstate, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            metric = traj_batch.info
            if config.get("DEBUG"):
                def callback(info):
                    average_price = info["average_price"][info["done"]]
                    timesteps = info["timestep"][info["done"]]
                    action_at_execution = info["action"][info["done"]]
                    for t in range(len(timesteps)):
                        print(f"agent=1, global step={timesteps[t]}, average price={average_price[t]}")
                        print(f"agent=1, global step={timesteps[t]}, action_at_execution={action_at_execution[t]}")
                jax.debug.callback(callback, metric)
            train_state = update_state[0]
            return train_state, metric
        self.update = _update

        def _reset():
            new_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], config["HIDDEN_SIZE"])
            return new_hstate
        self.reset = _reset
