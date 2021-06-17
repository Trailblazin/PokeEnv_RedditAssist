# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from poke_env.player_configuration import PlayerConfiguration
from poke_env.player.env_player import Gen7EnvSinglePlayer
from poke_env.player.random_player import RandomPlayer
from poke_env.server_configuration import LocalhostServerConfiguration

from rl.agents.ddpg import DDPGAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from tensorflow.keras.layers import Dense, Flatten,Input, Concatenate, Activation
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam

# We define our RL player
# It needs a state embedder and a reward computer, hence these two methods
class SimpleRLPlayer(Gen7EnvSinglePlayer):
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


class MaxDamagePlayer(RandomPlayer):
    def choose_move(self, battle):
        # If the player can attack, it will
        if battle.available_moves:
            # Finds the best move among available ones
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            return self.create_order(best_move)

        # If no attack is available, a random switch will be made
        else:
            return self.choose_random_move(battle)


NB_TRAINING_STEPS = 10000
NB_EVALUATION_EPISODES = 100

tf.random.set_seed(0)
np.random.seed(0)

# This is the function that will be used to train the dqn
def ddpg_training(player, ddpg, nb_steps):
    ddpg.fit(player, nb_steps=nb_steps)
    player.complete_current_battle()


def ddpg_evaluation(player, ddpg, nb_episodes):
    # Reset battle statistics
    player.reset_battles()
    ddpg.test(player, nb_episodes=nb_episodes, visualize=False, verbose=False)

    print(
        "DDPG Evaluation: %d victories out of %d episodes"
        % (player.n_won_battles, nb_episodes)
    )

if __name__ == "__main__":
    env_player = SimpleRLPlayer(
        player_configuration=PlayerConfiguration("RL Player", None),
        battle_format="gen7randombattle",
        server_configuration=LocalhostServerConfiguration,
    )

    opponent = RandomPlayer(
        player_configuration=PlayerConfiguration("Random player", None),
        battle_format="gen7randombattle",
        server_configuration=LocalhostServerConfiguration,
    )

    second_opponent = MaxDamagePlayer(
        player_configuration=PlayerConfiguration("Max damage player", None),
        battle_format="gen7randombattle",
        server_configuration=LocalhostServerConfiguration,
    )
    #print(env_player._observations)

#DDPG Model creation
 # Output dimension
    n_actions = len(env_player.action_space)

    actor = Sequential()
    actor.add(Dense(128, activation="relu", input_shape=(1, 10)))

    # Our embedding have shape (1, 10), which affects our hidden layer
    # dimension and output dimension
    # Flattening resolve potential issues that would arise otherwise
    actor.add(Flatten())
    actor.add(Dense(64, activation="relu"))
    actor.add(Dense(n_actions, activation="linear"))

    action_input = Input(shape=(n_actions))
    observation_input = Input(shape=(1,) + env_player._observations)
    flattened_observation = Flatten()(observation_input)

    x = Concatenate()([action_input, flattened_observation])
    x = Dense(64)(x)
    x = Activation('relu')(x)
    x = Dense(1)(x)
    x = Activation('linear')(x)
    critic = Model(inputs=[action_input, observation_input], outputs=x)

    memory = SequentialMemory(limit=10000, window_length=1)
    random_process = OrnsteinUhlenbeckProcess(size=18, theta=.15, mu=0., sigma=.3)
     # Simple epsilon greedy
    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr="eps",
        value_max=1.0,
        value_min=0.05,
        value_test=0,
        nb_steps=10000,
    )

    ddpg = DDPGAgent(
        actor=actor,
        critic=critic,
        critic_action_input=action_input,
        nb_actions=n_actions,
        policy=policy,
        memory=memory,
        nb_steps_warmup=1000,
        gamma=0.5,
        target_model_update=1,
        delta_clip=0.01,
    )
    ddpg.compile(Adam(lr=0.00025), metrics=["mae"])

        # Training
    env_player.play_against(
        env_algorithm=ddpg_training,
        opponent=opponent,
        env_algorithm_kwargs={"ddpg": ddpg, "nb_steps": NB_TRAINING_STEPS},
    )
    #ddpg.save("model_DDPG_%d" % NB_TRAINING_STEPS)
    #model.save('PokeEnvModels/DDPG_Model_%d' % NB_TRAINING_STEPS)

    # Evaluation
    print("Results against random player:")
    env_player.play_against(
        env_algorithm=ddpg_evaluation,
        opponent=opponent,
        env_algorithm_kwargs={"ddpg": ddpg, "nb_episodes": NB_EVALUATION_EPISODES},
    )

    print("\nResults against max player:")
    env_player.play_against(
        env_algorithm=ddpg_evaluation,
        opponent=second_opponent,
        env_algorithm_kwargs={"ddpg": ddpg, "nb_episodes": NB_EVALUATION_EPISODES},
    )
