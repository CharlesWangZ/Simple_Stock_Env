from typing import Optional, Tuple
from functools import partial
import chex
import jax
import jax.numpy as jnp
from gymnax.environments import environment, spaces



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
     time_to_execute: int = 100
     qty_to_execute: int = 100
     sigma: float = 0.2
     impact_factor: float = 0.1
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

        def get_obs(state: EnvState, params: EnvParams) -> Tuple[jnp.ndarray, jnp.ndarray]:
            obs_agent1 = jnp.array([state.stock_price - params.initial_stock_price, state.quant_remaining[0], jnp.array(state.time_left), state.revenue[0]])
            obs_agent2 = jnp.array([state.stock_price - params.initial_stock_price, state.quant_remaining[1], jnp.array(state.time_left), state.revenue[1]])
            return obs_agent1, obs_agent2

        def _step(
            key: chex.PRNGKey,
            state: EnvState,
            actions: Tuple[int, int],
            params: EnvParams,
        ):
            a1, a2 = actions
            a1 = jnp.where(state.time_left <= 0, state.quant_remaining[0], a1)
            a1 = jnp.clip(a1, 0, state.quant_remaining[0])
            a2 = jnp.where(state.time_left <= 0, state.quant_remaining[1], a2)
            a2 = jnp.clip(a2, 0, state.quant_remaining[1])
    
            d1 = jnp.logical_or(state.quant_remaining[0] - a1 <= 0, state.time_left <= 0)
            d2 = jnp.logical_or(state.quant_remaining[1] - a2 <= 0, state.time_left <= 0)
            all_done = jnp.logical_and(d1,d2)

            price = state.stock_price * jnp.exp(params.mu * params.dt + params.sigma * jnp.sqrt(params.dt) * jax.random.normal(key))
            price_impact = params.impact_factor * (a1 + a2)
            new_price = price - price_impact

            new_quant_remaining = state.quant_remaining - jnp.array(actions)
            new_revenue = state.revenue + new_price * jnp.array(actions)
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
            unused, state_reset = _reset(state, params)
            state_reset.dones = jnp.array([True, True])
            new_state = jax.tree_map(lambda x, y: jax.lax.select(all_done, x, y), state_reset, state_temp)

            (obs1, obs2) = get_obs(new_state, params)

            info1 = {"total_revenue": new_revenue[0],\
                     "quant_executed":params.qty_to_execute - new_quant_remaining[0],\
                     "done":d1,\
                     "all_done":all_done,\
                     "average_price":new_revenue[0]/(params.qty_to_execute - new_quant_remaining[0]),\
                     "timestep":state.step,\
                     }
            info2 = {"total_revenue": new_revenue[1],\
                     "quant_executed":params.qty_to_execute - new_quant_remaining[1],\
                     "done":d2,\
                     "all_done":all_done,\
                     "average_price":new_revenue[1]/(params.qty_to_execute - new_quant_remaining[1]),\
                     "timestep":state.step,\
                     }
            
            return (
                (obs1, obs2),
                new_state,
                (r1, r2),
                (d1,d2),
                (info1,info2),
            )

        def _reset(
            key: chex.PRNGKey, params: EnvParams
        ) -> Tuple[Tuple, EnvState]:
            state = EnvState(
                stock_price = 100.0,
                quant_remaining = jnp.full(2, 100),
                time_left = 100,
                revenue = jnp.full(2, 0.0),
                dones = jnp.full(2, False),
                step = 0,
            )
            (obs1, obs2) = get_obs(state, params)

            return (obs1, obs2), state

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
        return spaces.Discrete(100)

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

    env = Stock_GBM()
    env_params = env.default_params

    num_envs = 3
    initial_key = jax.random.PRNGKey(0)  
    rng = jax.random.split(initial_key, num_envs)  
    
    env.reset = jax.vmap(
        env.reset, in_axes=(0, None), out_axes=(0, 0)
    )

    observations, env_state = env.reset(rng, env_params)
    # print(observations, env_state)
    # init_x = jnp.zeros(env.observation_space(env_params).shape)
    print(env.action_space().n)
    print(env.observation_space(env_params).shape)



    






    

    num_envs = 100
    initial_key = jax.random.PRNGKey(0)  
    rng = jax.random.split(initial_key, num_envs)  

    env = Stock_GBM()
    env_params = env.default_params

    # action = jnp.ones((num_envs,), dtype=jnp.float32)
    action = jnp.full((num_envs,), 1)

    env.reset = jax.vmap(
        env.reset, in_axes=(0, None), out_axes=(0, 0)
    )

    env.step = jax.vmap(
        env.step,
        in_axes=(0, 0, 0, None),
        out_axes=(0, 0, 0, 0, 0),
    )
    

    obs, env_state = env.reset(rng, env_params)

    PRICE = []

    for i in range(100):
        actions = (action, action)
        initial_key = jax.random.PRNGKey(i)  
        rng = jax.random.split(initial_key, num_envs)
        obs, env_state, rewards, done, info = env.step(
            rng, env_state, actions, env_params
        )
        PRICE.append(env_state.stock_price)

        


    plt.plot(PRICE)

    plt.title("STOCK PRICE")
    plt.xlabel("time")
    plt.ylabel("price")

    # Show the plot
    plt.show()
    plt.savefig('ave_stock_prices_GBM_2.png')
        
