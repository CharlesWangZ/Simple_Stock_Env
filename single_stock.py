from typing import Optional, Tuple
from functools import partial
import chex
import jax
import jax.numpy as jnp
from gymnax.environments import environment, spaces
import matplotlib.pyplot as plt
from typing import Sequence, NamedTuple, Any, Dict


selling_quantity = 100
impact_factor = 10/selling_quantity

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
     time_to_execute: int = selling_quantity
     qty_to_execute: int = selling_quantity
     sigma: float = 0.2
     impact_factor: float = impact_factor
     mu: float = 0.0
     dt: float = 1/252
     max_time: int = 180000

class Stock_GBM(environment.Environment):
    env_id = "stock_gbm"

    """
    JAX Compatible version of geometric brownian motion stock environment.
    """

    def __init__(self):
        super().__init__()

        def get_obs(state: EnvState, params: EnvParams):
            obs_agent1 = jnp.array([state.stock_price - params.initial_stock_price, state.quant_remaining, jnp.array(state.time_left), state.revenue])
            return obs_agent1

        def _step(
            rng: chex.PRNGKey,
            state: EnvState,
            action: int,
            params: EnvParams,
        ):
            a1 = action
            a1 = jnp.where(state.time_left <= 0, state.quant_remaining, a1)
            a1 = jnp.clip(a1, 0, state.quant_remaining)

            # d1 = jnp.logical_or(state.quant_remaining - a1 <= 0, state.time_left <= 0)
            d1 = state.time_left <= 0
            
            rng, _rng = jax.random.split(rng)
            price = state.stock_price * jnp.exp(params.mu * params.dt + params.sigma * jnp.sqrt(params.dt) * jax.random.normal(_rng))
            price_impact = params.impact_factor * (a1 + 1)
            new_price = price - price_impact

            new_quant_remaining = state.quant_remaining - jnp.array(a1)
            new_revenue = state.revenue + new_price * jnp.array(a1)
            # average_price = new_revenue/(params.qty_to_execute - new_quant_remaining + 1e-8)
            # r1_temp = new_revenue/ (params.qty_to_execute - new_quant_remaining)/ (state.step + 1e-8)
            # r1 = jnp.where(d1, average_price, 0)
            r1 = (new_price - params.initial_stock_price) * a1

            state_temp = EnvState(
                stock_price=new_price,
                quant_remaining=new_quant_remaining,
                time_left=state.time_left - 1,
                revenue=new_revenue,
                dones=d1,
                step=state.step + 1
            )
            rng, _rng = jax.random.split(rng)
            unused, state_reset = _reset(_rng, params)
            state_reset.dones = True
            # new_state = jax.tree_map(lambda x, y: jax.lax.select(d1, x, y), state_reset, state_temp)
            new_state = jax.lax.cond(d1, 
                         lambda _: state_reset, 
                         lambda _: state_temp, 
                         operand=None)

            obs1 = get_obs(new_state, params)

            info1 = {"total_revenue": new_revenue,\
                     "quant_executed":params.qty_to_execute - new_quant_remaining,\
                     "done":d1,\
                     "average_price":new_revenue/(params.qty_to_execute - new_quant_remaining + 1e-8),\
                     "timestep":state.step,\
                     "Implementation_shortfall":((new_price - params.initial_stock_price) * a1),\
                     "action":a1,\
                     }
            
            return (
                obs1,
                new_state,
                r1,
                d1,
                info1,
            )

        def _reset(
            rng: chex.PRNGKey, params: EnvParams
        ) -> Tuple[Tuple, EnvState]:
            state = EnvState(
                stock_price = 100.0,
                quant_remaining = selling_quantity,
                time_left = selling_quantity - 1 ,
                revenue = 0.0,
                dones = False,
                step = 0,
            )
            obs = get_obs(state, params)

            return obs, state

        # overwrite Gymnax as it makes single-agent assumptions
        self.step = jax.jit(_step)
        self.reset = jax.jit(_reset)

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
        return spaces.Discrete(selling_quantity)

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


    rng = jax.random.PRNGKey(0)  
    env = Stock_GBM()
    env_params = env.default_params
    env.batch_reset = jax.vmap(env.reset, in_axes=(0, None))
    env.batch_step = jax.vmap(
        env.step, in_axes=(0, 0, 0, None)
    )
    
    num_env = 10
    num_steps = 100
    rng, _rng = jax.random.split(rng)
    rng_reset = jax.random.split(_rng, num_env)
    obsv, env_state = env.batch_reset(rng_reset, env_params)
        
    def _inner_rollout(carry, inner_step):
        (rng, observations, env_state, env_params) = carry
        rng, _rng = jax.random.split(rng)
        # actions = jax.random.randint(key=_rng, shape=(num_env,), minval=0, maxval=100)        
        actions = jnp.full(num_env, 1)
        actions = jnp.where(env_state.time_left <= 0, env_state.quant_remaining, actions)
        actions = jnp.clip(actions, 0, env_state.quant_remaining)

        rng, _rng = jax.random.split(rng)
        rng_step = jax.random.split(_rng, num_env)
        observations, env_state, rewards, dones, info = env.batch_step(rng_step, env_state, actions, env_params)
        # inner_timestep = jnp.full_like(rewards, inner_step, dtype=jnp.int32)
        traj = Transition(dones, actions, jnp.full(num_env,1), rewards, jnp.full(num_env,1), observations, info)
        return (rng, observations, env_state, env_params), (traj)
    rng, _rng = jax.random.split(rng)
    carry = (_rng, obsv, env_state, env_params)
    
    vals, traj_batch = jax.lax.scan(
    _inner_rollout,
    carry,
    None,
    length=num_steps,
    )




    info = traj_batch.info
    return_values = info["average_price"][info["done"]]
    timesteps = info["timestep"][info["done"]]
    action_at_execution = info["action"][info["done"]]
    for t in range(len(timesteps)):
        print(f"global step={timesteps[t]}, average price={return_values[t]}")
        print(f"global step={timesteps[t]}, action_at_execution={action_at_execution[t]}")

    print(info)
    print(traj_batch)


    # done_counts = traj_batch.info["done"].sum(axis=0)
    # print(done_counts)
    # reward = traj_batch.reward / done_counts
    # print(traj_batch.reward)
    # traj_batch = Transition(traj_batch.done, traj_batch.action, traj_batch.value, reward, traj_batch.log_prob, traj_batch.obs, traj_batch.info)
    # print(traj_batch.reward)


    

    
    # print(trajectories.info["average_price"][trajectories.info["done"]])
    # print(trajectories.info["action"])
    # print(trajectories.action)
    # print(trajectories.info["action"] - trajectories.action) #should be zero
    # print(trajectories.info["action"][trajectories.info["done"]])























    # # INPUT:
    # # rng = [1,
    # #        2,
    # #        3,
    # #        4]

    # # env_state = [state1,
    # #              state2,
    # #               state3,
    # #                state4 ]

    # # actions = ([1, [1,
    # #            1,   1, 
    # #            1,   1,    
    # #            1,]  1,])

    # # env_params = [params]


    # # OUTPUT:
    # # (obs,obs) = ([obs1, [obs1,
    # #            obs2,   obs2, 
    # #            obs3,   obs3,    
    # #            obs4,]  obs4,])

    # # new_state = [state1,
    # #              state2,
    # #               state3,
    # #                state4 ] 

    # # reward = ([r1, [r1,
    # #            r2,   r2, 
    # #            r3,   r3,    
    # #            r4,]  r4,])

    # # done = ([d1, [d1,
    # #            d2,   d2, 
    # #            d3,   d3,    
    # #            d4,]  d4,])

    # # info = ([i1, [i1,
    # #            i2,   i2, 
    # #            i3,   i3,    
    # #            i4,]  i4,])









