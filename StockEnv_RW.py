import jax
import jax.numpy as jnp
from typing import NamedTuple
from typing import Tuple
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
import chex
from jaxmarl.environments import spaces
from typing import Optional, Tuple
from flax import struct
from jax import random
from gymnax.environments import environment, spaces
import matplotlib.pyplot as plt



@struct.dataclass
class EnvState:
    stock_price: float
    quant_remaining:float
    time_left: int
    revenue: float

@struct.dataclass
class EnvParams:
    initial_stock_price: float = 100.0
    qty_to_execute: int = 100
    max_time: int = 5*60*60
    sigma: float = 0.2
    impact_factor: float = 0.1

    
class StockEnv_RW(environment.Environment):
    def __init__(self):
        super().__init__()
        self.obs_shape = (4,)

    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters
        return EnvParams()

    def step_env(
        self, key: chex.PRNGKey, state: EnvState, action: int, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        price_increments = params.sigma * jax.random.normal(key) 
        done = self.is_terminal(state,action,params)
        action = jnp.where(done, state.quant_remaining, action)
        action = jnp.clip(action, 0, state.quant_remaining) 
        price_impact = params.impact_factor * action * jax.random.uniform(key, minval=0, maxval=1)
        # price_impact = params.impact_factor * action 
        new_price = state.stock_price + price_increments - price_impact
        new_quant_remaining = state.quant_remaining - action
        new_revenue = state.revenue + new_price * action
    
        new_state = EnvState(new_price, new_quant_remaining, state.time_left - 1, new_revenue)

        return self.get_obs(new_state, params),new_state, ((new_price - params.initial_stock_price) * action) , done,\
              {"total_revenue":new_revenue,\
               "quant_executed":params.qty_to_execute - new_quant_remaining,\
               'done':done,\
               "average_price":new_revenue/(params.qty_to_execute - new_quant_remaining),\
                }

    def reset_env(self, key: chex.PRNGKey, params: EnvParams):
        state = EnvState(
            params.initial_stock_price,
            params.qty_to_execute,
            100,
            0.0
        )
        return self.get_obs(state, params), state
    
    def get_obs(self, state: EnvState, params: EnvParams) -> chex.Array:
        return jnp.array([state.stock_price - params.initial_stock_price, state.quant_remaining, state.time_left, state.revenue])
        # return jnp.array([state.stock_price - params.initial_stock_price, state.quant_remaining, state.time_left])
    
    @property
    def num_actions(self) -> int:
        return 1
    
    def is_terminal(self, state: EnvState, action, params: EnvParams) -> bool:
        return jnp.logical_or(
            state.time_left - 1 <= 0, state.quant_remaining - action <= 0
        )

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(100)
    
    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        high = jnp.array(
            [
                2000,
                params.qty_to_execute,
                params.max_time,
                100000,

            ]
        )
        return spaces.Box(0, high, (4,), dtype=jnp.float32)
    
    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        high = jnp.array(
            [
                2000,
                params.qty_to_execute,
                params.max_time,
                100000,
            ]
        )
        return spaces.Box(0, high, (4,), dtype=jnp.float32)
    
    @property
    def name(self) -> str:
        return "stock-v0"

if __name__ == "__main__":

    vmap_plot = True
    if vmap_plot:
        num_envs = 100
        num_steps = 100
        sell_all_qty = 100  

        with_agent_actions = jnp.full((num_steps,), 1)

        without_agent_actions = jnp.zeros((num_steps,))

        sell_all_at_once_actions = jnp.full((num_steps,), 2)

        def split_key(key):
            return jax.random.split(key,2)
        vmap_split = jax.vmap(split_key, in_axes=0, out_axes=0)

        def run_strategy_simulation(vmap_step, vmap_keys, initial_states, actions):
            stock_prices = jnp.zeros((num_envs, num_steps))

            for i in range(num_steps):
                keys = vmap_split(vmap_keys)
                vmap_keys = keys[:, 0]

                action = jnp.full((num_envs,), actions[i])  # Same action for all environments
                _, new_states, _, _, _ = vmap_step(vmap_keys, initial_states, action, env_params)
                stock_prices = stock_prices.at[:, i].set(new_states.stock_price)
                initial_states = new_states  # Update states for the next step

            return stock_prices

        # Reset the environments

        env = StockEnv_RW()
        env_params=env.default_params
        rng = jax.random.PRNGKey(60)

        vmap_reset = jax.vmap(env.reset_env, in_axes=(0, None))
        vmap_step = jax.vmap(env.step_env, in_axes=(0, 0, 0, None))
        
        vmap_keys = jax.random.split(rng, num_envs)
        obs, initial_states = vmap_reset(vmap_keys, env_params)

        # Run simulations for each strategy
        prices_with_agent = run_strategy_simulation(vmap_step, vmap_keys, initial_states, with_agent_actions)
        prices_without_agent = run_strategy_simulation(vmap_step, vmap_keys, initial_states, without_agent_actions)
        prices_sell_all_at_once = run_strategy_simulation(vmap_step, vmap_keys, initial_states, sell_all_at_once_actions)

        avg_with_agent = jnp.mean(prices_with_agent, axis=0)
        std_with_agent = jnp.std(prices_with_agent, axis=0)
        # print(avg_with_agent)

        avg_without_agent = jnp.mean(prices_without_agent, axis=0)
        std_without_agent = jnp.std(prices_without_agent, axis=0)
        # print(avg_without_agent)

        avg_sell_all_at_once = jnp.mean(prices_sell_all_at_once, axis=0)
        std_sell_all_at_once = jnp.std(prices_sell_all_at_once, axis=0)
        # print(avg_sell_all_at_once)

        time_steps = jnp.arange(num_steps)

        plt.figure(figsize=(12, 6))

        # With Agent Selling Stocks (mean line with shaded standard deviation)
        plt.plot(time_steps, avg_with_agent, label='With Agent', color='blue')
        plt.fill_between(time_steps, avg_with_agent - std_with_agent, avg_with_agent + std_with_agent, color='blue', alpha=0.1)

        # Without Agent Selling Stocks (mean line with shaded standard deviation)
        plt.plot(time_steps, avg_without_agent, label='Without Agent', color='orange')
        plt.fill_between(time_steps, avg_without_agent - std_without_agent, avg_without_agent + std_without_agent, color='orange', alpha=0.1)

        # Sell All at Once (mean line with shaded standard deviation)
        plt.plot(time_steps, avg_sell_all_at_once, label='Sell All at Once', color='green')
        plt.fill_between(time_steps, avg_sell_all_at_once - std_sell_all_at_once, avg_sell_all_at_once + std_sell_all_at_once, color='green', alpha=0.1)

        plt.title('Average Stock Price Over Time with Standard Deviation')
        plt.xlabel('Time Steps')
        plt.ylabel('Average Stock Price')
        plt.legend()
        plt.show()
        plt.savefig('ave_stock_prices_RW_1000.png')




    enable_vmap= False
    if enable_vmap:
        env = StockEnv_RW()
        env_params=env.default_params
        rng = jax.random.PRNGKey(10)
        # rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)

        vmap_reset = jax.vmap(env.reset, in_axes=(0, None))
        vmap_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))
        vmap_act_sample=jax.vmap(env.action_space().sample, in_axes=(0))
        num_envs = 10
        vmap_keys = jax.random.split(rng, num_envs)
        obs, state = vmap_reset(vmap_keys, env_params)
        # test_action0 = jnp.array([0] * num_envs)
        # test_action1 = jnp.array([10] * num_envs)
        # test_action2 = jnp.array([10] * num_envs)

        

        for i in range(200):
            test_actions=vmap_act_sample(vmap_keys)
            print(test_actions)
            obs, state, reward, done, _ = vmap_step(vmap_keys, state, test_actions, env_params)
            print(obs)

        
            
            



            
            
