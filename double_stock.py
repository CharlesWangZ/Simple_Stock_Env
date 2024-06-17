from typing import Optional, Tuple
from functools import partial
import chex
import jax
import jax.numpy as jnp
from gymnax.environments import environment, spaces
import matplotlib.pyplot as plt
from typing import Sequence, NamedTuple, Any
import flax.linen as nn
import optax
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
import distrax
import numpy as np


qty_to_execute = 10
impact_factor = 10/qty_to_execute

@chex.dataclass
class EnvState:
    stock_price: float  
    quant_remaining: chex.Array 
    time_left: int
    revenue: chex.Array 
    dones: chex.Array 
    step: int

@chex.dataclass
class EnvParams:
     initial_stock_price: float = 100.0
     time_to_execute: int = qty_to_execute
     qty_to_execute: int = qty_to_execute
     sigma: float = 0.0
     impact_factor: float = impact_factor
     mu: float = 0.0
     dt: float = 1/252
     max_time: int = 180000


class Stock_GBM_MULTI(environment.Environment):
    env_id = "stock_gbm"

    """
    JAX Compatible version of geometric brownian motion stock environment.
    """

    def __init__(self):
        super().__init__()

        def get_obs(state: EnvState, params: EnvParams) -> Tuple[jnp.ndarray, jnp.ndarray]:
            obs_agent1 = jnp.array([state.stock_price - params.initial_stock_price, state.quant_remaining[0], jnp.array(state.time_left), state.revenue[0]])
            obs_agent2 = jnp.array([state.stock_price - params.initial_stock_price, state.quant_remaining[1], jnp.array(state.time_left), state.revenue[1]])
            return obs_agent1, obs_agent2

        def _step(
            rng: chex.PRNGKey,
            state: EnvState,
            actions: Tuple[int, int],
            params: EnvParams,
        ):
            action_1, action_2 = actions
            a1 = action_1

            action_1 = jnp.where(state.time_left <= 0, state.quant_remaining[0], action_1)
            a1 = jnp.clip(action_1, 0, state.quant_remaining[0])
            action_2 = jnp.where(state.time_left <= 0, state.quant_remaining[1], action_2)
            a2 = jnp.clip(action_2, 0, state.quant_remaining[1])

            d1 = jnp.logical_or(state.quant_remaining[0] - a1 <= 0, state.time_left <= 0)
            d2 = jnp.logical_or(state.quant_remaining[1] - a2 <= 0, state.time_left <= 0)
            all_done = jnp.logical_and(d1,d2)
            
            rng, _rng = jax.random.split(rng)
            price = state.stock_price * jnp.exp(params.mu * params.dt + params.sigma * jnp.sqrt(params.dt) * jax.random.normal(_rng))
            price_impact = params.impact_factor * (a1 + a2)
            new_price = price - price_impact

            new_quant_remaining = state.quant_remaining - jnp.array([a1, a2])
            new_revenue = state.revenue + new_price * jnp.array([a1, a2])
            r1 = (new_price - params.initial_stock_price) * a1 
            r2 = (new_price - params.initial_stock_price) * a2

            state_temp = EnvState(
                stock_price=new_price,
                quant_remaining=new_quant_remaining,
                time_left=state.time_left - 1,
                revenue=new_revenue,
                dones=jnp.array([d1, d2]),
                step=state.step + 1
            )
            rng, _rng = jax.random.split(rng)
            unused, state_reset = _reset(_rng, params)
            state_reset.dones = jnp.array([True, True])
            # new_state = jax.lax.select(all_done, state_reset, state_temp)
            # new_state = jax.tree_map(lambda x, y: jax.lax.select(all_done, x, y), state_reset, state_temp)
            new_state = jax.lax.cond(all_done, 
                         lambda _: state_reset, 
                         lambda _: state_temp, 
                         operand=None)

            (obs1, obs2) = get_obs(new_state, params)

            info1 = {"total_revenue": new_revenue[0],\
                    "quant_executed":params.qty_to_execute - new_quant_remaining[0],\
                    "done":d1,\
                    "all_done":all_done,\
                    "average_price":new_revenue[0]/(params.qty_to_execute - new_quant_remaining[0]),\
                    "Implementation_shortfall":((new_price - params.initial_stock_price) * a1),\
                    "timestep":state.step,\
                    "action":a1,\
                    }
            info2 = {"total_revenue": new_revenue[1],\
                    "quant_executed":params.qty_to_execute - new_quant_remaining[1],\
                    "done":d2,\
                    "all_done":all_done,\
                    "average_price":new_revenue[1]/(params.qty_to_execute - new_quant_remaining[1]),\
                    "Implementation_shortfall":((new_price - params.initial_stock_price) * a2),\
                    "timestep":state.step,\
                    "action":a2,\
                    }
            
            return (
                (obs1, obs2),
                new_state,
                (r1, r2),
                (d1,d2),
                (info1, info2),
            )

        def _reset(
            key: chex.PRNGKey, params: EnvParams
        ) -> Tuple[Tuple, EnvState]:
            state = EnvState(
                stock_price = 100.0,
                quant_remaining = jnp.full(2, qty_to_execute),
                time_left = qty_to_execute-1,
                revenue = jnp.full(2, 0.0),
                dones = jnp.full(2, False),
                step = 0,
            )
            (obs1, obs2) = get_obs(state, params)

            return (obs1, obs2), state

        # overwrite Gymnax as it makes single-agent assumptions
        # self.step = jax.jit(_step)
        # self.reset = jax.jit(_reset)
        self.step = _step
        self.reset = _reset

    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters
        return EnvParams()

    @property
    def name(self) -> str:
        """Environment name."""
        return "stock_gbm"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 1

    def action_space(
        self, params: Optional[EnvParams] = None
    ) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(qty_to_execute)

    def observation_space(self, params: EnvParams) -> spaces.Discrete:
        """Observation space of the environment."""
        low = jnp.array(
            [
                -100,
                0,
                0,
                0,
            ]
        )
        high = jnp.array(
            [
                0,                      # Upper bound for the first observation
                params.qty_to_execute,  # Upper bounds for the rest of the observations
                params.max_time,
                100000,
            ]
        )
        return spaces.Box(low, high, (4,), dtype=jnp.float32)

    def state_space(self, params: EnvParams) -> spaces.Discrete:
        """State space of the environment."""
        return spaces.Discrete(5)
    



if __name__ == "__main__":

    class Transition(NamedTuple):
        done: jnp.ndarray
        action: jnp.ndarray
        value: jnp.ndarray
        reward: jnp.ndarray
        log_prob: jnp.ndarray
        obs: jnp.ndarray
        info: jnp.ndarray


    rng = jax.random.PRNGKey(110)  
    env = Stock_GBM_MULTI()
    env_params = env.default_params
    env.batch_reset = jax.vmap(env.reset, in_axes=(0, None))
    env.batch_step = jax.vmap(
        env.step, in_axes=(0, 0, 0, None)
    )
    
    num_env = 3
    rng, _rng = jax.random.split(rng)
    rng_reset = jax.random.split(_rng, num_env)
    obsv, env_state = env.batch_reset(rng_reset, env_params)
    # print(obsv)
    # print("obs1",obsv[0])
    # print(env_state.quant_remaining)

    def _inner_rollout(carry, inner_step):
        (rng, obsv, env_state, env_params) = carry

        rng, _rng = jax.random.split(rng)
        action_1 = jax.random.randint(key=_rng, shape=(num_env,), minval=0, maxval=20)
        quant_remaining_1 = env_state.quant_remaining[:, 0]
        action_1 = jnp.where(env_state.time_left <= 0, quant_remaining_1, action_1)
        action_1 = jnp.clip(action_1, 0, quant_remaining_1)

        rng, _rng = jax.random.split(rng)
        action_2 = action_1 * 2
        quant_remaining_2 = env_state.quant_remaining[:, 1]
        action_2 = jnp.where(env_state.time_left <= 0, quant_remaining_2, action_2)
        action_2 = jnp.clip(action_2, 0, quant_remaining_2)

        rng, _rng = jax.random.split(rng)
        rng_step = jax.random.split(_rng, num_env)
        actions = (action_1, action_2)
        obsv, env_state, rewards, dones, info = env.batch_step(rng_step, env_state, actions, env_params)

        traj1 = Transition(dones[0], action_1, jnp.full(num_env,1), rewards[0], jnp.full(num_env,1), obsv[0], info[0])
        traj2 = Transition(dones[1], action_2, None, rewards[1], None, obsv[1], info[1])
        # inner_timestep = jnp.full_like(rewards, inner_step, dtype=jnp.int32)
        # traj = Transition(dones, actions, jnp.full((num_env,2),1), rewards, jnp.full((num_env,2),1), observations, info)
        return (rng, obsv, env_state, env_params), (traj1,traj2)
    
    rng, _rng = jax.random.split(rng)
    carry = (_rng, obsv, env_state, env_params)
    vals, traj_batch = jax.lax.scan(
    _inner_rollout,
    carry,
    None,
    length=20,
    )

    # done_counts_1 = traj_batch[0].info["done"].sum(axis=0)
    # reward_1 = traj_batch[0].reward / done_counts_1
    # traj_batch_1 = Transition(traj_batch[0].done, traj_batch[0].action, traj_batch[0].value, reward_1, traj_batch[0].log_prob, traj_batch[0].obs, traj_batch[0].info)
    # done_counts_2 = traj_batch[1].info["done"].sum(axis=0)
    # reward_2 = traj_batch[1].reward / done_counts_2
    # traj_batch_2 = Transition(traj_batch[1].done, traj_batch[1].action, traj_batch[1].value, reward_2, traj_batch[1].log_prob, traj_batch[1].obs, traj_batch[1].info)
    # print(traj_batch[0].reward)
    # traj_batch = (traj_batch_1, traj_batch_2)
    # print(traj_batch[0].reward)



    traj1 = traj_batch[0]
    print(traj1.info)
    # print(traj1.done)
    # print(traj1.reward)
    

    # traj2 = traj_batch[1]
    # print(traj1.info["average_price"][traj1.info["all_done"]])
    # # # print(trajectories.info["action"])
    # # # print(trajectories.action)
    # # print(traj1.info["action"] - traj1.action) #should be zero
    # # print(trajectories.info["action"][trajectories.info["done"]])
    # actions_1 = traj1.info["action"].reshape(-1)
    # np.savetxt("actions_1.csv", actions_1, delimiter=",", header="Action", comments='', fmt='%d')
    # actions_2 = traj2.info["action"].reshape(-1)
    # np.savetxt("actions_2.csv", actions_2, delimiter=",", header="Action", comments='', fmt='%d')
    
    # print("quant1:",traj1.obs[:,:,1])
    # print("quant2:",traj2.obs[:,:,1])




    # for i in range(50):
    #     rng, _rng = jax.random.split(rng)
    #     action_1 = jax.random.randint(key=_rng, shape=(num_env,), minval=0, maxval=100)
    #     quant_remaining_1 = env_state.quant_remaining[:, 0]
    #     action_1 = jnp.where(env_state.time_left <= 0, quant_remaining_1, action_1)
    #     action_1 = jnp.clip(action_1, 0, quant_remaining_1)
        
    #     rng, _rng = jax.random.split(rng)
    #     action_2 = jnp.full(num_env, 1)
    #     quant_remaining_2 = env_state.quant_remaining[:, 1]
    #     action_2 = jnp.where(env_state.time_left <= 0, quant_remaining_2, action_2)
    #     action_2 = jnp.clip(action_2, 0, quant_remaining_2)
    #     # print("quant_remaining_2:",quant_remaining_2)
        

    #     rng, _rng = jax.random.split(rng)
    #     rng_step = jax.random.split(_rng, num_env)
    #     actions = (action_1, action_2)
    #     # print("actions:",actions)
    #     print("action_1:",action_1)
    #     print("action_2:",action_2)
        
        
    #     obsv, env_state, rewards, dones, info = env.batch_step(rng_step, env_state, actions, env_params)


    #     print("quant_remaining:",env_state.quant_remaining)
    #     print("averages prices:",info[0]["average_price"],info[1]["average_price"])
    #     print("actions in info:",info[0]["action"],info[1]["action"])
    #     print("--------")





    # INPUT:
    # rng = [1,
    #        2,
    #        3,
    #        4]

    # env_state = [state1,
    #              state2,
    #               state3,
    #                state4 ]

    # actions = ([1, [1,
    #            1,   1, 
    #            1,   1,    
    #            1,]  1,])

    # env_params = [params]


    # OUTPUT:
    # (obs,obs) = ([obs1, [obs1,
    #            obs2,   obs2, 
    #            obs3,   obs3,    
    #            obs4,]  obs4,])

    # new_state = [state1,
    #              state2,
    #               state3,
    #                state4 ] 

    # reward = ([r1, [r1,
    #            r2,   r2, 
    #            r3,   r3,    
    #            r4,]  r4,])

    # done = ([d1, [d1,
    #            d2,   d2, 
    #            d3,   d3,    
    #            d4,]  d4,])

    # info = ([i1, [i1,
    #            i2,   i2, 
    #            i3,   i3,    
    #            i4,]  i4,])









