import gym
from stable_baselines3 import PPO, DQN

import asyncio
import numpy as np
from gym.spaces import Box, Discrete

from poke_env.player.env_player import Gen8EnvSinglePlayer
from poke_env.player.random_player import RandomPlayer


class SimpleRLPlayer(Gen8EnvSinglePlayer):

    observation_space = Box(low=-10, high=10, shape=(10,))
    action_space = Discrete(22)

    def getThisPlayer(self):
        return self

    def __init__(self, *args, **kwargs):
        Gen8EnvSinglePlayer.__init__(self)

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





envPlayer = SimpleRLPlayer()
opponent = RandomPlayer()


model = DQN('MlpPolicy', envPlayer, gamma=0.5, verbose=1)
def ray_training_function(player):

    print ("Training...")
    model.learn(total_timesteps=1000)
    print("Training complete.")


def ray_evaluating_function(player):
    player.reset_battles()
    for _ in range(100):
        done = False
        obs = player.reset()
        while not done:
            action = model.predict(obs)[0]
            obs, _, done, _ = player.step(action)
            # print ("done:" + str(done))
    player.complete_current_battle()

    print(
        "DQN Evaluation: %d victories out of %d episodes"
        % (player.n_won_battles, 100)
    )

# Training
envPlayer.play_against(
    env_algorithm=ray_training_function,
    opponent=opponent,
)

envPlayer.play_against(
    env_algorithm=ray_evaluating_function,
    opponent=opponent,
)

model.save("TrainedAgents/DQN/vsRandom/DQNvsRandom")