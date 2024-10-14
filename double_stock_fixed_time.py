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
    last_action: chex.Array

@chex.dataclass
class EnvParams:
     initial_stock_price: float = 100.0
     time_to_execute: int = qty_to_execute
     qty_to_execute: int = qty_to_execute
     sigma: float = 0.0
     impact_factor: float = impact_factor
     mu: float = 0.0
     dt: float = 1/252 # number of working days in a year
    
class Stock_GBM_MULTI(environment.Environment):
    env_id = "stock_gbm"
    """
    JAX Compatible version of geometric brownian motion stock environment.
    Each agent gets observation (stock price, quant_remaining, time_left, and its revenue) and decides next action size
    """
    def __init__(self):
        super().__init__()
        def get_obs(state: EnvState, params: EnvParams):
            obs_agent1 = jnp.array([state.stock_price - params.initial_stock_price, state.quant_remaining[0], jnp.array(state.time_left), state.revenue[0], state.last_action[0]])
            obs_agent2 = jnp.array([state.stock_price - params.initial_stock_price, state.quant_remaining[1], jnp.array(state.time_left), state.revenue[1], state.last_action[1]])
            return obs_agent1, obs_agent2

        def _step(
            rng: chex.PRNGKey,
            state: EnvState,
            actions: Tuple[int, int],
            params: EnvParams,
        ):
            action_1, action_2 = actions
            # enforce an agent to sell not more than its inventory
            action_1 = jnp.where(state.time_left <= 0, state.quant_remaining[0], action_1)
            a1 = jnp.clip(action_1, 0, state.quant_remaining[0])
            action_2 = jnp.where(state.time_left <= 0, state.quant_remaining[1], action_2)
            a2 = jnp.clip(action_2, 0, state.quant_remaining[1])
            done = state.time_left <= 0

            rng, _rng = jax.random.split(rng)
            price = state.stock_price * jnp.exp(params.mu * params.dt + params.sigma * jnp.sqrt(params.dt) * jax.random.normal(_rng))
            # Assume linear price impact
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
                dones=jnp.array([done, done]),
                step=state.step + 1,
                last_action=jnp.array([a2, 0])
            )
            rng, _rng = jax.random.split(rng)
            unused, state_reset = _reset(_rng, params)
            state_reset.dones = jnp.array([True, True])
            new_state = jax.lax.cond(done, 
                         lambda _: state_reset, 
                         lambda _: state_temp, 
                         operand=None)

            (obs1, obs2) = get_obs(new_state, params)

            info1 = {"total_revenue": new_revenue[0],\
                    "quant_executed":params.qty_to_execute - new_quant_remaining[0],\
                    "done":done,\
                    "average_price":new_revenue[0]/(params.qty_to_execute - new_quant_remaining[0]),\
                    "Implementation_shortfall":((new_price - params.initial_stock_price) * a1),\
                    "timestep":state.step,\
                    "action":a1,\
                    "opponent_action": a2,\
                    }
            info2 = {"total_revenue": new_revenue[1],\
                    "quant_executed":params.qty_to_execute - new_quant_remaining[1],\
                    "done":done,\
                    "average_price":new_revenue[1]/(params.qty_to_execute - new_quant_remaining[1]),\
                    "Implementation_shortfall":((new_price - params.initial_stock_price) * a2),\
                    "timestep":state.step,\
                    "action":a2,\
                    "opponent_action": a1,\
                    }
            
            return (
                (obs1, obs2),
                new_state,
                (r1, r2),
                (done,done),
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
                last_action = jnp.full(2, 0),
            )
            (obs1, obs2) = get_obs(state, params)

            return (obs1, obs2), state

        # overwrite Gymnax as it makes single-agent assumptions
        self.step = jax.jit(_step)
        self.reset = jax.jit(_reset)
        # self.step = _step
        # self.reset = _reset

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
                -100, # Lower bound for the observations
                0,
                0,
                0,
                0,
            ]
        )
        high = jnp.array(
            [
                0,         # Upper bounds for the observations             
                params.qty_to_execute,  
                params.max_time,
                100000,
                100,
            ]
        )
        return spaces.Box(low, high, (5,), dtype=jnp.float32)

    def state_space(self, params: EnvParams) -> spaces.Discrete:
        """State space of the environment."""
        return spaces.Discrete(6)
    


