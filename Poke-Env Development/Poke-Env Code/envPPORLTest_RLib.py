import asyncio
import numpy as np
import ray
import ray.rllib.agents.ppo as ppo


from asyncio import ensure_future, new_event_loop, set_event_loop
from gym.spaces import Box, Discrete
from poke_env.player.env_player import Gen8EnvSinglePlayer
from poke_env.player.random_player import RandomPlayer


class SimpleRLPlayer(Gen8EnvSinglePlayer):
    def __init__(self, *args, **kwargs):
        Gen8EnvSinglePlayer.__init__(self)
        self.observation_space = Box(low=-10, high=10, shape=(10,))

    @property
    def action_space(self):
        return Discrete(22)

    def embed_battle(self, battle):
        # -1 indicates that the move does not have a base power
        # or is not available
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = (
                move.base_power / 100
            )  # Simple rescaling to facilitate learning
            if move.type:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                )

        # We count how many pokemons have not fainted in each team
        remaining_mon_team = (
            len([mon for mon in battle.team.values() if mon.fainted]) / 6
        )
        remaining_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        )

        # Final vector with 10 components
        return np.concatenate(
            [
                moves_base_power,
                moves_dmg_multiplier,
                [remaining_mon_team, remaining_mon_opponent],
            ]
        )

    def compute_reward(self, battle) -> float:
        return self.reward_computing_helper(
            battle, fainted_value=2, hp_value=1, victory_value=30
        )

    def observation_space(self):
        return np.array


ray.init()
config = ppo.DEFAULT_CONFIG.copy()
config["num_gpus"] = 0
config["num_workers"] = 0  # Training will not work with poke-env if this value != 0
config["framework"] = "tfe"
config["gamma"] = 0.7 #Test and tinker with this?!

trainer = ppo.PPOTrainer(config=config, env=SimpleRLPlayer)

def ray_training_function(player):
    for i in range(2):
        result = trainer.train()
        print(result)
    player.complete_current_battle()

env_player = trainer.workers.local_worker().env
opponent = RandomPlayer()

env_player.play_against(
    env_algorithm=ray_training_function,
    opponent=opponent,
)
def ray_evaluating_function(player):
    player.reset_battles()
    for _ in range(100):
        done = False
        obs = player.reset()
        while not done:
            action = trainer.compute_action(obs)
            obs, _, done, _ = player.step(action)
    player.complete_current_battle()

    print(
        "PPO Evaluation: %d victories out of %d episodes"
        % (player.n_won_battles, 100)
    )

env_player = trainer.workers.local_worker().env
opponent = RandomPlayer()

#training
env_player.play_against(
    env_algorithm=ray_training_function,
    opponent=opponent,
)

#evaluating
env_player.play_against(
    env_algorithm=ray_evaluating_function,
    opponent=opponent,
)