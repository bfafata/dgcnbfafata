import os
import shutil

import numpy as np
import pandas as pd
import torch as T
from scipy.special import softmax
# from maddpgv2_gnn.maddpgv2_gnn import maddpgv2_gnn
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data

# from maddpg_gnn.maddpg_gnn import maddpg_gnn
from mappo_gnn import mappo_gnn
from utils.utils import make_env, complete_graph_edge_index, update_noise_exponential_decay, calculate_elo_rating, \
    update_agent_goals_softmax_weights, update_adver_goals_softmax_weights

# general options
SCENARIO_NAME = "zone_def_tag"
AGENT_MODEL = "mappo_gnn"
ADVER_MODEL = "mappo_gnn"
AGENT_MODE = "train"
ADVER_MODE = "train"
GENERAL_TRAINING_NAME = SCENARIO_NAME + "_agent_" + AGENT_MODEL + "_vs_opp_" + ADVER_MODEL + "_1_vs_1_30_time_steps"
AGENT_TRAINING_NAME = GENERAL_TRAINING_NAME + "_agent"
ADVER_TRAINING_NAME = GENERAL_TRAINING_NAME + "_adver"
TENSORBOARD_LOG_DIRECTORY = "tensorboard_log" + '/' + SCENARIO_NAME + '/' + GENERAL_TRAINING_NAME
CSV_LOG_DIRECTORY = "csv_log" + '/' + SCENARIO_NAME
NUMBER_OF_EPISODES = 500000
EPISODE_TIME_STEP_LIMIT = 30
RENDER_ENV = True
SAVE_MODEL_RATE = 100
SAVE_CSV_LOG = True

# elo options
INITIAL_ELO = 1000.0
ELO_DIFFRENCE = 1000.0
ELO_D = 400
RESULTS_REWARD_DICT = {'1': [0, 1], '2': [1, 0], '3': [0.5, 0.5], '4': [0.5, 0.5]}

AGENT_TASK_DIFFICULTY_COEFFICIENT = 1
ADVER_TASK_DIFFICULTY_COEFFICIENT = 1
AGENT_ELO_K = 32 * AGENT_TASK_DIFFICULTY_COEFFICIENT
ADVER_ELO_K = 32 * ADVER_TASK_DIFFICULTY_COEFFICIENT

# env and drone options
POSITION_DIMENSIONS = 2
COMMUNICATION_DIMENSIONS = 1
NUMBER_OF_AGENT_DRONES = 1
NUMBER_OF_ADVER_DRONES = 1
NUMBER_OF_LANDMARKS = 0
RESTRICTED_RADIUS = 0.2
INTERCEPT_RADIUS = 0.7
RADAR_NOISE_POSITION = 0.1
RADAR_NOISE_VELOCITY = 0.5
BIG_REWARD_CONSTANT = 10.0
REWARD_MULTIPLIER_CONSTANT = 2.0
LANDMARK_SIZE = 0.05
EXPONENTIAL_NOISE_DECAY = True
EXPONENTIAL_NOISE_DECAY_CONSTANT = 0.0002
EXIT_SCREEN_TERMINATE = True

AGENT_DRONE_RADIUS = 0.25
AGENT_DRONE_SIZE = 0.075
AGENT_DRONE_DENSITY = 25.0
AGENT_DRONE_INITIAL_MASS = 1.0
AGENT_DRONE_ACCEL = 4.0
AGENT_DRONE_MAX_SPEED = 1.0
AGENT_DRONE_COLLIDE = True
AGENT_DRONE_SILENT = False
AGENT_DRONE_U_NOISE = 1.0
AGENT_DRONE_C_NOISE = 0.5
AGENT_DRONE_U_RANGE = 1.0

ADVER_DRONE_RADIUS = 0.25
ADVER_DRONE_SIZE = 0.075
ADVER_DRONE_DENSITY = 25.0
ADVER_DRONE_INITIAL_MASS = 1.0
ADVER_DRONE_ACCEL = 4.0
ADVER_DRONE_MAX_SPEED = 1.0
ADVER_DRONE_COLLIDE = True
ADVER_DRONE_SILENT = False
ADVER_DRONE_U_NOISE = 1.0
ADVER_DRONE_C_NOISE = 0.5
ADVER_DRONE_U_RANGE = 1.0

# maddpg options for agent
AGENT_MADDPG_GNN_DISCOUNT_RATE = 0.99
AGENT_MADDPG_GNN_LEARNING_RATE_ACTOR = 0.0005
AGENT_MADDPG_GNN_LEARNING_RATE_CRITIC = 0.0005
AGENT_MADDPG_GNN_ACTOR_DROPOUT = 0
AGENT_MADDPG_GNN_CRITIC_DROPOUT = 0
AGENT_MADDPG_GNN_TAU = 0.01
AGENT_MADDPG_GNN_MEMORY_SIZE = 10000
AGENT_MADDPG_GNN_BATCH_SIZE = 128
AGENT_MADDPG_GNN_UPDATE_TARGET = None
AGENT_MADDPG_GNN_GRADIENT_CLIPPING = True
AGENT_MADDPG_GNN_GRADIENT_NORM_CLIP = 1

AGENT_MADDPG_GNN_ACTOR_INPUT_DIMENSIONS = [(1 + NUMBER_OF_LANDMARKS * POSITION_DIMENSIONS + (
        NUMBER_OF_AGENT_DRONES + NUMBER_OF_ADVER_DRONES) * (POSITION_DIMENSIONS * 2
                                                            + COMMUNICATION_DIMENSIONS)) if SCENARIO_NAME == "zone_def_push" else (
        1 + NUMBER_OF_LANDMARKS * POSITION_DIMENSIONS +
        (NUMBER_OF_AGENT_DRONES + NUMBER_OF_ADVER_DRONES) * (
                POSITION_DIMENSIONS * 2 + COMMUNICATION_DIMENSIONS) + NUMBER_OF_AGENT_DRONES +
        NUMBER_OF_ADVER_DRONES) for i in range(NUMBER_OF_AGENT_DRONES)]
AGENT_MADDPG_GNN_ACTOR_OUTPUT_DIMENSIONS = [128, 128, 128]
AGENT_MADDPG_GNN_U_ACTIONS_DIMENSIONS = POSITION_DIMENSIONS
AGENT_MADDPG_GNN_C_ACTIONS_DIMENSIONS = COMMUNICATION_DIMENSIONS
AGENT_MADDPG_GNN_ACTIONS_DIMENSIONS = AGENT_MADDPG_GNN_U_ACTIONS_DIMENSIONS + AGENT_MADDPG_GNN_C_ACTIONS_DIMENSIONS

AGENT_MADDPG_GNN_CRITIC_GNN_INPUT_DIMS = [(1 + NUMBER_OF_LANDMARKS * POSITION_DIMENSIONS + (
        NUMBER_OF_AGENT_DRONES + NUMBER_OF_ADVER_DRONES) * (POSITION_DIMENSIONS * 2
                                                            + COMMUNICATION_DIMENSIONS)) if SCENARIO_NAME == "zone_def_push" else (
        1 + NUMBER_OF_LANDMARKS * POSITION_DIMENSIONS +
        (NUMBER_OF_AGENT_DRONES + NUMBER_OF_ADVER_DRONES) * (
                POSITION_DIMENSIONS * 2 + COMMUNICATION_DIMENSIONS) + NUMBER_OF_AGENT_DRONES +
        NUMBER_OF_ADVER_DRONES) for i in range(NUMBER_OF_AGENT_DRONES)]
AGENT_MADDPG_GNN_CRITIC_GNN_NUM_HEADS = 1
AGENT_MADDPG_GNN_CRITIC_BOOL_CONCAT = True
AGENT_MADDPG_GNN_CRITIC_GNN_OUTPUT_DIMS = [128, 128, 128]

AGENT_MADDPG_GNN_CRITIC_GMT_HIDDEN_DIMS = 128
AGENT_MADDPG_GNN_CRITIC_GMT_OUTPUT_DIMS = 128

AGENT_MADDPG_GNN_CRITIC_U_ACTIONS_FC_INPUT_DIMS = AGENT_MADDPG_GNN_U_ACTIONS_DIMENSIONS * NUMBER_OF_AGENT_DRONES
AGENT_MADDPG_GNN_CRITIC_C_ACTIONS_FC_INPUT_DIMS = AGENT_MADDPG_GNN_C_ACTIONS_DIMENSIONS * NUMBER_OF_AGENT_DRONES
AGENT_MADDPG_GNN_CRITIC_U_ACTIONS_FC_OUTPUT_DIMS = [64, 64]
AGENT_MADDPG_GNN_CRITIC_C_ACTIONS_FC_OUTPUT_DIMS = [64, 64]
AGENT_MADDPG_GNN_CRITIC_CONCAT_FC_OUTPUT_DIMS = [128, 128]

# maddpg options for adversary
ADVER_MADDPG_GNN_DISCOUNT_RATE = 0.99
ADVER_MADDPG_GNN_LEARNING_RATE_ACTOR = 0.0005
ADVER_MADDPG_GNN_LEARNING_RATE_CRITIC = 0.0005
ADVER_MADDPG_GNN_ACTOR_DROPOUT = 0
ADVER_MADDPG_GNN_CRITIC_DROPOUT = 0
ADVER_MADDPG_GNN_TAU = 0.01
ADVER_MADDPG_GNN_MEMORY_SIZE = 10000
ADVER_MADDPG_GNN_BATCH_SIZE = 128
ADVER_MADDPG_GNN_UPDATE_TARGET = None
ADVER_MADDPG_GNN_GRADIENT_CLIPPING = True
ADVER_MADDPG_GNN_GRADIENT_NORM_CLIP = 1

ADVER_MADDPG_GNN_ACTOR_INPUT_DIMENSIONS = [(1 + NUMBER_OF_LANDMARKS * POSITION_DIMENSIONS + (
        NUMBER_OF_AGENT_DRONES + NUMBER_OF_ADVER_DRONES) * (POSITION_DIMENSIONS * 2
                                                            + COMMUNICATION_DIMENSIONS)) if SCENARIO_NAME == "zone_def_push" else (
        1 + NUMBER_OF_LANDMARKS * POSITION_DIMENSIONS +
        (NUMBER_OF_AGENT_DRONES + NUMBER_OF_ADVER_DRONES) * (
                POSITION_DIMENSIONS * 2 + COMMUNICATION_DIMENSIONS) + NUMBER_OF_AGENT_DRONES +
        NUMBER_OF_ADVER_DRONES) for i in range(NUMBER_OF_ADVER_DRONES)]
ADVER_MADDPG_GNN_ACTOR_OUTPUT_DIMENSIONS = [128, 128, 128]
ADVER_MADDPG_GNN_U_ACTIONS_DIMENSIONS = POSITION_DIMENSIONS
ADVER_MADDPG_GNN_C_ACTIONS_DIMENSIONS = COMMUNICATION_DIMENSIONS
ADVER_MADDPG_GNN_ACTIONS_DIMENSIONS = ADVER_MADDPG_GNN_U_ACTIONS_DIMENSIONS + ADVER_MADDPG_GNN_C_ACTIONS_DIMENSIONS

ADVER_MADDPG_GNN_CRITIC_GNN_INPUT_DIMS = [(1 + NUMBER_OF_LANDMARKS * POSITION_DIMENSIONS + (
        NUMBER_OF_AGENT_DRONES + NUMBER_OF_ADVER_DRONES) * (POSITION_DIMENSIONS * 2
                                                            + COMMUNICATION_DIMENSIONS)) if SCENARIO_NAME == "zone_def_push" else (
        1 + NUMBER_OF_LANDMARKS * POSITION_DIMENSIONS +
        (NUMBER_OF_AGENT_DRONES + NUMBER_OF_ADVER_DRONES) * (
                POSITION_DIMENSIONS * 2 + COMMUNICATION_DIMENSIONS) + NUMBER_OF_AGENT_DRONES +
        NUMBER_OF_ADVER_DRONES) for i in range(NUMBER_OF_ADVER_DRONES)]
ADVER_MADDPG_GNN_CRITIC_GNN_NUM_HEADS = 1
ADVER_MADDPG_GNN_CRITIC_BOOL_CONCAT = True
ADVER_MADDPG_GNN_CRITIC_GNN_OUTPUT_DIMS = [128, 128, 128]

ADVER_MADDPG_GNN_CRITIC_GMT_HIDDEN_DIMS = 128
ADVER_MADDPG_GNN_CRITIC_GMT_OUTPUT_DIMS = 128

ADVER_MADDPG_GNN_CRITIC_U_ACTIONS_FC_INPUT_DIMS = ADVER_MADDPG_GNN_U_ACTIONS_DIMENSIONS * NUMBER_OF_ADVER_DRONES
ADVER_MADDPG_GNN_CRITIC_C_ACTIONS_FC_INPUT_DIMS = ADVER_MADDPG_GNN_C_ACTIONS_DIMENSIONS * NUMBER_OF_ADVER_DRONES
ADVER_MADDPG_GNN_CRITIC_U_ACTIONS_FC_OUTPUT_DIMS = [64, 64]
ADVER_MADDPG_GNN_CRITIC_C_ACTIONS_FC_OUTPUT_DIMS = [64, 64]
ADVER_MADDPG_GNN_CRITIC_CONCAT_FC_OUTPUT_DIMS = [128, 128]

# mappo options for agent
AGENT_MAPPO_GNN_LEARNING_RATE_ACTOR = 0.0005
AGENT_MAPPO_GNN_LEARNING_RATE_CRITIC = 0.0005
AGENT_MAPPO_GNN_ACTOR_DROPOUT = 0
AGENT_MAPPO_GNN_CRITIC_DROPOUT = 0
AGENT_MAPPO_GNN_BATCH_SIZE = 20
AGENT_MAPPO_GNN_GAMMA = 0.99
AGENT_MAPPO_GNN_CLIP_COEFFICIENT = 0.2
AGENT_MAPPO_GNN_NUMBER_OF_EPOCHS = 5
AGENT_MAPPO_GNN_GAE_LAMBDA = 0.95
AGENT_MAPPO_GNN_ENTROPY_COEFFICIENT = 0.01
AGENT_MAPPO_GNN_USE_HUBER_LOSS = True
AGENT_MAPPO_GNN_HUBER_DELTA = 10.0
AGENT_MAPPO_GNN_USE_CLIPPED_VALUE_LOSS = True
AGENT_MAPPO_GNN_CRITIC_LOSS_COEFFICIENT = 0.5
AGENT_MAPPO_GNN_GRADIENT_CLIPPING = True
AGENT_MAPPO_GNN_GRADIENT_NORM_CLIP = 1
AGENT_MAPPO_GNN_EPISODE_LENGTH = AGENT_MAPPO_GNN_BATCH_SIZE * AGENT_MAPPO_GNN_NUMBER_OF_EPOCHS

AGENT_MAPPO_GNN_ACTOR_INPUT_DIMENSIONS = [(1 + NUMBER_OF_LANDMARKS * POSITION_DIMENSIONS + (
        NUMBER_OF_AGENT_DRONES + NUMBER_OF_ADVER_DRONES) * (POSITION_DIMENSIONS * 2
                                                            + COMMUNICATION_DIMENSIONS)) if SCENARIO_NAME == "zone_def_push" else (
        1 + NUMBER_OF_LANDMARKS * POSITION_DIMENSIONS +
        (NUMBER_OF_AGENT_DRONES + NUMBER_OF_ADVER_DRONES) * (
                POSITION_DIMENSIONS * 2 + COMMUNICATION_DIMENSIONS) + NUMBER_OF_AGENT_DRONES +
        NUMBER_OF_ADVER_DRONES) for i in range(NUMBER_OF_AGENT_DRONES)]
AGENT_MAPPO_GNN_ACTOR_OUTPUT_DIMENSIONS = [128, 128, 128]
AGENT_MAPPO_GNN_U_ACTIONS_DIMENSIONS = POSITION_DIMENSIONS
AGENT_MAPPO_GNN_C_ACTIONS_DIMENSIONS = COMMUNICATION_DIMENSIONS
AGENT_MAPPO_GNN_ACTIONS_DIMENSIONS = AGENT_MAPPO_GNN_U_ACTIONS_DIMENSIONS + AGENT_MAPPO_GNN_C_ACTIONS_DIMENSIONS

AGENT_MAPPO_GNN_CRITIC_GNN_INPUT_DIMS = [(1 + NUMBER_OF_LANDMARKS * POSITION_DIMENSIONS + (
        NUMBER_OF_AGENT_DRONES + NUMBER_OF_ADVER_DRONES) * (POSITION_DIMENSIONS * 2
                                                            + COMMUNICATION_DIMENSIONS)) if SCENARIO_NAME == "zone_def_push" else (
        1 + NUMBER_OF_LANDMARKS * POSITION_DIMENSIONS +
        (NUMBER_OF_AGENT_DRONES + NUMBER_OF_ADVER_DRONES) * (
                POSITION_DIMENSIONS * 2 + COMMUNICATION_DIMENSIONS) + NUMBER_OF_AGENT_DRONES +
        NUMBER_OF_ADVER_DRONES) for i in range(NUMBER_OF_AGENT_DRONES)]
AGENT_MAPPO_GNN_CRITIC_GNN_NUM_HEADS = 1
AGENT_MAPPO_GNN_CRITIC_BOOL_CONCAT = True
AGENT_MAPPO_GNN_CRITIC_GNN_OUTPUT_DIMS = [128, 128, 128]

AGENT_MAPPO_GNN_CRITIC_GMT_HIDDEN_DIMS = 128
AGENT_MAPPO_GNN_CRITIC_GMT_OUTPUT_DIMS = 128
AGENT_MAPPO_GNN_CRITIC_FC_OUTPUT_DIMS = [128, 128]

# mappo options for adver
ADVER_MAPPO_GNN_LEARNING_RATE_ACTOR = 0.0005
ADVER_MAPPO_GNN_LEARNING_RATE_CRITIC = 0.0005
ADVER_MAPPO_GNN_ACTOR_DROPOUT = 0
ADVER_MAPPO_GNN_CRITIC_DROPOUT = 0
ADVER_MAPPO_GNN_BATCH_SIZE = 20
ADVER_MAPPO_GNN_GAMMA = 0.99
ADVER_MAPPO_GNN_CLIP_COEFFICIENT = 0.2
ADVER_MAPPO_GNN_NUMBER_OF_EPOCHS = 5
ADVER_MAPPO_GNN_GAE_LAMBDA = 0.95
ADVER_MAPPO_GNN_ENTROPY_COEFFICIENT = 0.01
ADVER_MAPPO_GNN_USE_HUBER_LOSS = True
ADVER_MAPPO_GNN_HUBER_DELTA = 10.0
ADVER_MAPPO_GNN_USE_CLIPPED_VALUE_LOSS = True
ADVER_MAPPO_GNN_CRITIC_LOSS_COEFFICIENT = 0.5
ADVER_MAPPO_GNN_GRADIENT_CLIPPING = True
ADVER_MAPPO_GNN_GRADIENT_NORM_CLIP = 1
ADVER_MAPPO_GNN_EPISODE_LENGTH = ADVER_MAPPO_GNN_BATCH_SIZE * ADVER_MAPPO_GNN_NUMBER_OF_EPOCHS

ADVER_MAPPO_GNN_ACTOR_INPUT_DIMENSIONS = [(1 + NUMBER_OF_LANDMARKS * POSITION_DIMENSIONS + (
        NUMBER_OF_AGENT_DRONES + NUMBER_OF_ADVER_DRONES) * (POSITION_DIMENSIONS * 2
                                                            + COMMUNICATION_DIMENSIONS)) if SCENARIO_NAME == "zone_def_push" else (
        1 + NUMBER_OF_LANDMARKS * POSITION_DIMENSIONS +
        (NUMBER_OF_AGENT_DRONES + NUMBER_OF_ADVER_DRONES) * (
                POSITION_DIMENSIONS * 2 + COMMUNICATION_DIMENSIONS) + NUMBER_OF_AGENT_DRONES +
        NUMBER_OF_ADVER_DRONES) for i in range(NUMBER_OF_ADVER_DRONES)]
ADVER_MAPPO_GNN_ACTOR_OUTPUT_DIMENSIONS = [128, 128, 128]
ADVER_MAPPO_GNN_U_ACTIONS_DIMENSIONS = POSITION_DIMENSIONS
ADVER_MAPPO_GNN_C_ACTIONS_DIMENSIONS = COMMUNICATION_DIMENSIONS
ADVER_MAPPO_GNN_ACTIONS_DIMENSIONS = ADVER_MAPPO_GNN_U_ACTIONS_DIMENSIONS + ADVER_MAPPO_GNN_C_ACTIONS_DIMENSIONS

ADVER_MAPPO_GNN_CRITIC_GNN_INPUT_DIMS = [(1 + NUMBER_OF_LANDMARKS * POSITION_DIMENSIONS + (
        NUMBER_OF_AGENT_DRONES + NUMBER_OF_ADVER_DRONES) * (POSITION_DIMENSIONS * 2
                                                            + COMMUNICATION_DIMENSIONS)) if SCENARIO_NAME == "zone_def_push" else (
        1 + NUMBER_OF_LANDMARKS * POSITION_DIMENSIONS +
        (NUMBER_OF_AGENT_DRONES + NUMBER_OF_ADVER_DRONES) * (
                POSITION_DIMENSIONS * 2 + COMMUNICATION_DIMENSIONS) + NUMBER_OF_AGENT_DRONES +
        NUMBER_OF_ADVER_DRONES) for i in range(NUMBER_OF_ADVER_DRONES)]
ADVER_MAPPO_GNN_CRITIC_GNN_NUM_HEADS = 1
ADVER_MAPPO_GNN_CRITIC_BOOL_CONCAT = True
ADVER_MAPPO_GNN_CRITIC_GNN_OUTPUT_DIMS = [128, 128, 128]

ADVER_MAPPO_GNN_CRITIC_GMT_HIDDEN_DIMS = 128
ADVER_MAPPO_GNN_CRITIC_GMT_OUTPUT_DIMS = 128
ADVER_MAPPO_GNN_CRITIC_FC_OUTPUT_DIMS = [128, 128]

# maddpgv2 options for agent
AGENT_MADDPGV2_GNN_DISCOUNT_RATE = 0.99
AGENT_MADDPGV2_GNN_LEARNING_RATE_ACTOR = 0.0005
AGENT_MADDPGV2_GNN_LEARNING_RATE_CRITIC = 0.0005
AGENT_MADDPGV2_GNN_ACTOR_DROPOUT = 0
AGENT_MADDPGV2_GNN_CRITIC_DROPOUT = 0
AGENT_MADDPGV2_GNN_TAU = 0.01
AGENT_MADDPGV2_GNN_MEMORY_SIZE = 100000
AGENT_MADDPGV2_GNN_BATCH_SIZE = 128
AGENT_MADDPGV2_GNN_UPDATE_TARGET = None
AGENT_MADDPGV2_GNN_GRADIENT_CLIPPING = True
AGENT_MADDPGV2_GNN_GRADIENT_NORM_CLIP = 1
AGENT_MADDPGV2_GNN_GOAL = EPISODE_TIME_STEP_LIMIT
AGENT_MADDPGV2_GNN_NUMBER_OF_GOALS = 4
AGENT_MADDPGV2_GNN_GOAL_DIFFERENCE = 2
AGENT_MADDPGV2_GNN_GOAL_DISTRIBUTION = [AGENT_MADDPGV2_GNN_GOAL + i * AGENT_MADDPGV2_GNN_GOAL_DIFFERENCE for i in
                                        range(- AGENT_MADDPGV2_GNN_NUMBER_OF_GOALS + 1, 1)]
AGENT_MADDPGV2_GNN_ADDITIONAL_GOALS = 4
AGENT_MADDPGV2_GNN_GOAL_STRATEGY = "goal_distribution_v2"

AGENT_MADDPGV2_GNN_ACTOR_INPUT_DIMENSIONS = [(1 + NUMBER_OF_LANDMARKS * POSITION_DIMENSIONS + (
        NUMBER_OF_AGENT_DRONES + NUMBER_OF_ADVER_DRONES) * (POSITION_DIMENSIONS * 2
                                                            + COMMUNICATION_DIMENSIONS)) if SCENARIO_NAME == "zone_def_push" else (
        1 + NUMBER_OF_LANDMARKS * POSITION_DIMENSIONS +
        (NUMBER_OF_AGENT_DRONES + NUMBER_OF_ADVER_DRONES) * (
                POSITION_DIMENSIONS * 2 + COMMUNICATION_DIMENSIONS) + NUMBER_OF_AGENT_DRONES +
        NUMBER_OF_ADVER_DRONES) for i in range(NUMBER_OF_AGENT_DRONES)]
AGENT_MADDPGV2_GNN_ACTOR_OUTPUT_DIMENSIONS = [128, 128, 128]
AGENT_MADDPGV2_GNN_U_ACTIONS_DIMENSIONS = POSITION_DIMENSIONS
AGENT_MADDPGV2_GNN_C_ACTIONS_DIMENSIONS = COMMUNICATION_DIMENSIONS
AGENT_MADDPGV2_GNN_ACTIONS_DIMENSIONS = AGENT_MADDPGV2_GNN_U_ACTIONS_DIMENSIONS + AGENT_MADDPGV2_GNN_C_ACTIONS_DIMENSIONS

AGENT_MADDPGV2_GNN_CRITIC_GNN_INPUT_DIMS = [(1 + NUMBER_OF_LANDMARKS * POSITION_DIMENSIONS + (
        NUMBER_OF_AGENT_DRONES + NUMBER_OF_ADVER_DRONES) * (POSITION_DIMENSIONS * 2
                                                            + COMMUNICATION_DIMENSIONS)) if SCENARIO_NAME == "zone_def_push" else (
        1 + NUMBER_OF_LANDMARKS * POSITION_DIMENSIONS +
        (NUMBER_OF_AGENT_DRONES + NUMBER_OF_ADVER_DRONES) * (
                POSITION_DIMENSIONS * 2 + COMMUNICATION_DIMENSIONS) + NUMBER_OF_AGENT_DRONES +
        NUMBER_OF_ADVER_DRONES) for i in range(NUMBER_OF_AGENT_DRONES)]
AGENT_MADDPGV2_GNN_CRITIC_GNN_NUM_HEADS = 1
AGENT_MADDPGV2_GNN_CRITIC_BOOL_CONCAT = True
AGENT_MADDPGV2_GNN_CRITIC_GNN_OUTPUT_DIMS = [128, 128, 128]

AGENT_MADDPGV2_GNN_CRITIC_GMT_HIDDEN_DIMS = 128
AGENT_MADDPGV2_GNN_CRITIC_GMT_OUTPUT_DIMS = 128

AGENT_MADDPGV2_GNN_CRITIC_U_ACTIONS_FC_INPUT_DIMS = AGENT_MADDPGV2_GNN_U_ACTIONS_DIMENSIONS * NUMBER_OF_AGENT_DRONES
AGENT_MADDPGV2_GNN_CRITIC_C_ACTIONS_FC_INPUT_DIMS = AGENT_MADDPGV2_GNN_C_ACTIONS_DIMENSIONS * NUMBER_OF_AGENT_DRONES
AGENT_MADDPGV2_GNN_CRITIC_U_ACTIONS_FC_OUTPUT_DIMS = [64, 64]
AGENT_MADDPGV2_GNN_CRITIC_C_ACTIONS_FC_OUTPUT_DIMS = [64, 64]

AGENT_MADDPGV2_GNN_CRITIC_GOAL_FC_INPUT_DIMS = len([AGENT_MADDPGV2_GNN_GOAL])
AGENT_MADDPGV2_GNN_CRITIC_GOAL_FC_OUTPUT_DIMS = [8, 8]

AGENT_MADDPGV2_GNN_CRITIC_CONCAT_FC_OUTPUT_DIMS = [128, 128]

# maddpgv2 options for adversary
ADVER_MADDPGV2_GNN_DISCOUNT_RATE = 0.99
ADVER_MADDPGV2_GNN_LEARNING_RATE_ACTOR = 0.0005
ADVER_MADDPGV2_GNN_LEARNING_RATE_CRITIC = 0.0005
ADVER_MADDPGV2_GNN_ACTOR_DROPOUT = 0
ADVER_MADDPGV2_GNN_CRITIC_DROPOUT = 0
ADVER_MADDPGV2_GNN_TAU = 0.01
ADVER_MADDPGV2_GNN_MEMORY_SIZE = 100000
ADVER_MADDPGV2_GNN_BATCH_SIZE = 128
ADVER_MADDPGV2_GNN_UPDATE_TARGET = None
ADVER_MADDPGV2_GNN_GRADIENT_CLIPPING = True
ADVER_MADDPGV2_GNN_GRADIENT_NORM_CLIP = 1
ADVER_MADDPGV2_GNN_GOAL = RESTRICTED_RADIUS
ADVER_MADDPGV2_GNN_NUMBER_OF_GOALS = 5
ADVER_MADDPGV2_GNN_GOAL_DIFFERENCE = 0.025
ADVER_MADDPGV2_GNN_GOAL_DISTRIBUTION = [ADVER_MADDPGV2_GNN_GOAL + i * ADVER_MADDPGV2_GNN_GOAL_DIFFERENCE for i in
                                        range(ADVER_MADDPGV2_GNN_NUMBER_OF_GOALS)]
ADVER_MADDPGV2_GNN_ADDITIONAL_GOALS = 0
ADVER_MADDPGV2_GNN_GOAL_STRATEGY = "goal_distribution_v2"

ADVER_MADDPGV2_GNN_ACTOR_INPUT_DIMENSIONS = [(1 + NUMBER_OF_LANDMARKS * POSITION_DIMENSIONS + (
        NUMBER_OF_AGENT_DRONES + NUMBER_OF_ADVER_DRONES) * (POSITION_DIMENSIONS * 2
                                                            + COMMUNICATION_DIMENSIONS)) if SCENARIO_NAME == "zone_def_push" else (
        1 + NUMBER_OF_LANDMARKS * POSITION_DIMENSIONS +
        (NUMBER_OF_AGENT_DRONES + NUMBER_OF_ADVER_DRONES) * (
                POSITION_DIMENSIONS * 2 + COMMUNICATION_DIMENSIONS) + NUMBER_OF_AGENT_DRONES +
        NUMBER_OF_ADVER_DRONES) for i in range(NUMBER_OF_ADVER_DRONES)]
ADVER_MADDPGV2_GNN_ACTOR_OUTPUT_DIMENSIONS = [128, 128, 128]
ADVER_MADDPGV2_GNN_U_ACTIONS_DIMENSIONS = POSITION_DIMENSIONS
ADVER_MADDPGV2_GNN_C_ACTIONS_DIMENSIONS = COMMUNICATION_DIMENSIONS
ADVER_MADDPGV2_GNN_ACTIONS_DIMENSIONS = ADVER_MADDPG_GNN_U_ACTIONS_DIMENSIONS + ADVER_MADDPG_GNN_C_ACTIONS_DIMENSIONS

ADVER_MADDPGV2_GNN_CRITIC_GNN_INPUT_DIMS = [(1 + NUMBER_OF_LANDMARKS * POSITION_DIMENSIONS + (
        NUMBER_OF_AGENT_DRONES + NUMBER_OF_ADVER_DRONES) * (POSITION_DIMENSIONS * 2
                                                            + COMMUNICATION_DIMENSIONS)) if SCENARIO_NAME == "zone_def_push" else (
        1 + NUMBER_OF_LANDMARKS * POSITION_DIMENSIONS +
        (NUMBER_OF_AGENT_DRONES + NUMBER_OF_ADVER_DRONES) * (
                POSITION_DIMENSIONS * 2 + COMMUNICATION_DIMENSIONS) + NUMBER_OF_AGENT_DRONES +
        NUMBER_OF_ADVER_DRONES) for i in range(NUMBER_OF_ADVER_DRONES)]
ADVER_MADDPGV2_GNN_CRITIC_GNN_NUM_HEADS = 1
ADVER_MADDPGV2_GNN_CRITIC_BOOL_CONCAT = True
ADVER_MADDPGV2_GNN_CRITIC_GNN_OUTPUT_DIMS = [128, 128, 128]

ADVER_MADDPGV2_GNN_CRITIC_GMT_HIDDEN_DIMS = 128
ADVER_MADDPGV2_GNN_CRITIC_GMT_OUTPUT_DIMS = 128

ADVER_MADDPGV2_GNN_CRITIC_U_ACTIONS_FC_INPUT_DIMS = ADVER_MADDPG_GNN_U_ACTIONS_DIMENSIONS * NUMBER_OF_ADVER_DRONES
ADVER_MADDPGV2_GNN_CRITIC_C_ACTIONS_FC_INPUT_DIMS = ADVER_MADDPG_GNN_C_ACTIONS_DIMENSIONS * NUMBER_OF_ADVER_DRONES
ADVER_MADDPGV2_GNN_CRITIC_U_ACTIONS_FC_OUTPUT_DIMS = [64, 64]
ADVER_MADDPGV2_GNN_CRITIC_C_ACTIONS_FC_OUTPUT_DIMS = [64, 64]

ADVER_MADDPGV2_GNN_CRITIC_GOAL_FC_INPUT_DIMS = len([ADVER_MADDPGV2_GNN_GOAL])
ADVER_MADDPGV2_GNN_CRITIC_GOAL_FC_OUTPUT_DIMS = [8, 8]

ADVER_MADDPGV2_GNN_CRITIC_CONCAT_FC_OUTPUT_DIMS = [128, 128]


# Options for MAGCN
OPTIMIZER = "adam"
LR_SCHEDULER = "cosine_annealing_with_warm_restarts"
OBS_DIMS = 32
DGCN_OUTPUT_DIMS = 32
SOMU_LSTM_HIDDEN_SIZE = 128
SOMU_LSTM_NUM_LAYERS = 16
SOMU_LSTM_DROPOUT = 0.2
NUM_SOMU_LSTM = 8
SCMU_LSTM_HIDDEN_SIZE = 128
SCMU_LSTM_NUM_LAYERS = 8
SCMU_LSTM_DROPOUT = 0.2
NUM_SCMU_LSTM = 8
SOMU_MULTI_ATT_NUM_HEADS = 8
SOMU_MULTI_ATT_DROPOUT = 0.2
SCMU_MULTI_ATT_NUM_HEADS = 8
SCMU_MULTI_ATT_DROPOUT = 0.2
ACTOR_FC_OUTPUT_DIMS = 32
ACTOR_FC_DROPOUT_P = 0.2
SOFTMAX_ACTIONS_DIMS = 32
SOFTMAX_ACTIONS_DROPOUT_P = 0.2

def train_test():
    """ function to execute experiments to train or test models based on different algorithms """

    # check agent model
    if AGENT_MODEL == "maddpg_gnn":
        pass
        # generate maddpg gnn agents for agent drones
        # agent_maddpg_gnn_agents = maddpg_gnn(mode=AGENT_MODE, scenario_name=SCENARIO_NAME,
        #                                      training_name=AGENT_TRAINING_NAME,
        #                                      discount_rate=AGENT_MADDPG_GNN_DISCOUNT_RATE,
        #                                      lr_actor=AGENT_MADDPG_GNN_LEARNING_RATE_ACTOR,
        #                                      lr_critic=AGENT_MADDPG_GNN_LEARNING_RATE_CRITIC,
        #                                      num_agents=NUMBER_OF_AGENT_DRONES,
        #                                      num_opp=NUMBER_OF_ADVER_DRONES,
        #                                      actor_dropout_p=AGENT_MADDPG_GNN_ACTOR_DROPOUT,
        #                                      critic_dropout_p=AGENT_MADDPG_GNN_CRITIC_DROPOUT,
        #                                      state_fc_input_dims=AGENT_MADDPG_GNN_ACTOR_INPUT_DIMENSIONS,
        #                                      state_fc_output_dims=AGENT_MADDPG_GNN_ACTOR_OUTPUT_DIMENSIONS,
        #                                      u_action_dims=AGENT_MADDPG_GNN_U_ACTIONS_DIMENSIONS,
        #                                      c_action_dims=AGENT_MADDPG_GNN_C_ACTIONS_DIMENSIONS,
        #                                      num_heads=AGENT_MADDPG_GNN_CRITIC_GNN_NUM_HEADS,
        #                                      bool_concat=AGENT_MADDPG_GNN_CRITIC_BOOL_CONCAT,
        #                                      gnn_input_dims=AGENT_MADDPG_GNN_CRITIC_GNN_INPUT_DIMS,
        #                                      gnn_output_dims=AGENT_MADDPG_GNN_CRITIC_GNN_INPUT_DIMS,
        #                                      gmt_hidden_dims=AGENT_MADDPG_GNN_CRITIC_GMT_HIDDEN_DIMS,
        #                                      gmt_output_dims=AGENT_MADDPG_GNN_CRITIC_GMT_OUTPUT_DIMS,
        #                                      u_actions_fc_input_dims=AGENT_MADDPG_GNN_CRITIC_U_ACTIONS_FC_INPUT_DIMS,
        #                                      u_actions_fc_output_dims=AGENT_MADDPG_GNN_CRITIC_U_ACTIONS_FC_OUTPUT_DIMS,
        #                                      c_actions_fc_input_dims=AGENT_MADDPG_GNN_CRITIC_C_ACTIONS_FC_INPUT_DIMS,
        #                                      c_actions_fc_output_dims=AGENT_MADDPG_GNN_CRITIC_C_ACTIONS_FC_OUTPUT_DIMS,
        #                                      concat_fc_output_dims=AGENT_MADDPG_GNN_CRITIC_CONCAT_FC_OUTPUT_DIMS,
        #                                      tau=AGENT_MADDPG_GNN_TAU, mem_size=AGENT_MADDPG_GNN_MEMORY_SIZE,
        #                                      batch_size=AGENT_MADDPG_GNN_BATCH_SIZE,
        #                                      update_target=AGENT_MADDPG_GNN_UPDATE_TARGET,
        #                                      grad_clipping=AGENT_MADDPG_GNN_GRADIENT_CLIPPING,
        #                                      grad_norm_clip=AGENT_MADDPG_GNN_GRADIENT_NORM_CLIP, is_adversary=False)

    elif AGENT_MODEL == "mappo_gnn":

        # generate mappo gnn agents for agent drones
        agent_mappo_gnn_agents = mappo_gnn(mode=AGENT_MODE, scenario_name=SCENARIO_NAME,
                                           training_name=AGENT_TRAINING_NAME,
                                           lr_actor=AGENT_MAPPO_GNN_LEARNING_RATE_ACTOR,
                                           lr_critic=AGENT_MAPPO_GNN_LEARNING_RATE_CRITIC,
                                           num_agents=NUMBER_OF_AGENT_DRONES, num_opp=NUMBER_OF_ADVER_DRONES,
                                           u_range=AGENT_DRONE_U_RANGE,
                                           u_noise=AGENT_DRONE_U_NOISE, c_noise=AGENT_DRONE_C_NOISE, is_adversary=False,
                                           actor_dropout_p=AGENT_MAPPO_GNN_ACTOR_DROPOUT,
                                           critic_dropout_p=AGENT_MAPPO_GNN_CRITIC_DROPOUT,
                                           state_fc_input_dims=AGENT_MAPPO_GNN_ACTOR_INPUT_DIMENSIONS,
                                           state_fc_output_dims=AGENT_MAPPO_GNN_ACTOR_OUTPUT_DIMENSIONS,
                                           u_action_dims=AGENT_MAPPO_GNN_U_ACTIONS_DIMENSIONS,
                                           c_action_dims=AGENT_MAPPO_GNN_C_ACTIONS_DIMENSIONS,
                                           num_heads=AGENT_MAPPO_GNN_CRITIC_GNN_NUM_HEADS,
                                           bool_concat=AGENT_MAPPO_GNN_CRITIC_BOOL_CONCAT,
                                           gnn_input_dims=AGENT_MAPPO_GNN_CRITIC_GNN_INPUT_DIMS,
                                           gnn_output_dims=AGENT_MAPPO_GNN_CRITIC_GNN_OUTPUT_DIMS,
                                           gmt_hidden_dims=AGENT_MAPPO_GNN_CRITIC_GMT_HIDDEN_DIMS,
                                           gmt_output_dims=AGENT_MAPPO_GNN_CRITIC_GMT_OUTPUT_DIMS,
                                           fc_output_dims=AGENT_MAPPO_GNN_CRITIC_FC_OUTPUT_DIMS,
                                           batch_size=AGENT_MAPPO_GNN_BATCH_SIZE, gamma=AGENT_MAPPO_GNN_GAMMA,
                                           clip_coeff=AGENT_MAPPO_GNN_CLIP_COEFFICIENT,
                                           num_epochs=AGENT_MAPPO_GNN_NUMBER_OF_EPOCHS,
                                           gae_lambda=AGENT_MAPPO_GNN_GAE_LAMBDA,
                                           entropy_coeff=AGENT_MAPPO_GNN_ENTROPY_COEFFICIENT,
                                           use_huber_loss=AGENT_MAPPO_GNN_USE_HUBER_LOSS,
                                           huber_delta=AGENT_MAPPO_GNN_HUBER_DELTA,
                                           use_clipped_value_loss=AGENT_MAPPO_GNN_USE_CLIPPED_VALUE_LOSS,
                                           critic_loss_coeff=AGENT_MAPPO_GNN_CRITIC_LOSS_COEFFICIENT,
                                           grad_clipping=AGENT_MAPPO_GNN_GRADIENT_CLIPPING,
                                           grad_norm_clip=AGENT_MAPPO_GNN_GRADIENT_NORM_CLIP,
                                           optimizer=OPTIMIZER, lr_scheduler=LR_SCHEDULER, obs_dims=OBS_DIMS,
                                           dgcn_output_dims=DGCN_OUTPUT_DIMS,
                                           somu_lstm_hidden_size=SOMU_LSTM_HIDDEN_SIZE,
                                           somu_lstm_num_layers=SOMU_LSTM_NUM_LAYERS,
                                           somu_lstm_dropout=SOMU_LSTM_DROPOUT, num_somu_lstm=NUM_SOMU_LSTM,
                                           scmu_lstm_hidden_size=SCMU_LSTM_HIDDEN_SIZE,
                                           scmu_lstm_num_layers=SCMU_LSTM_NUM_LAYERS,
                                           scmu_lstm_dropout=SCMU_LSTM_DROPOUT,
                                           num_scmu_lstm=NUM_SCMU_LSTM,
                                           somu_multi_att_num_heads=SOMU_MULTI_ATT_NUM_HEADS,
                                           somu_multi_att_dropout=SOMU_MULTI_ATT_DROPOUT,
                                           scmu_multi_att_num_heads=SCMU_MULTI_ATT_NUM_HEADS,
                                           scmu_multi_att_dropout=SCMU_MULTI_ATT_DROPOUT,
                                           actor_fc_output_dims=ACTOR_FC_OUTPUT_DIMS,
                                           actor_fc_dropout_p=ACTOR_FC_DROPOUT_P,
                                           softmax_actions_dims=SOFTMAX_ACTIONS_DIMS,
                                           softmax_actions_dropout_p=SOFTMAX_ACTIONS_DROPOUT_P
                                           )

    elif AGENT_MODEL == "maddpgv2_gnn":
        pass
        # generate maddpgv2 gnn agents for agent drones
        # agent_maddpgv2_gnn_agents = maddpgv2_gnn(mode=AGENT_MODE, scenario_name=SCENARIO_NAME,
        #                                          training_name=AGENT_TRAINING_NAME,
        #                                          discount_rate=AGENT_MADDPGV2_GNN_DISCOUNT_RATE,
        #                                          lr_actor=AGENT_MADDPGV2_GNN_LEARNING_RATE_ACTOR,
        #                                          lr_critic=AGENT_MADDPGV2_GNN_LEARNING_RATE_CRITIC,
        #                                          num_agents=NUMBER_OF_AGENT_DRONES,
        #                                          num_opp=NUMBER_OF_ADVER_DRONES,
        #                                          actor_dropout_p=AGENT_MADDPGV2_GNN_ACTOR_DROPOUT,
        #                                          critic_dropout_p=AGENT_MADDPGV2_GNN_CRITIC_DROPOUT,
        #                                          state_fc_input_dims=AGENT_MADDPGV2_GNN_ACTOR_INPUT_DIMENSIONS,
        #                                          state_fc_output_dims=AGENT_MADDPGV2_GNN_ACTOR_OUTPUT_DIMENSIONS,
        #                                          u_action_dims=AGENT_MADDPGV2_GNN_U_ACTIONS_DIMENSIONS,
        #                                          c_action_dims=AGENT_MADDPGV2_GNN_C_ACTIONS_DIMENSIONS,
        #                                          num_heads=AGENT_MADDPGV2_GNN_CRITIC_GNN_NUM_HEADS,
        #                                          bool_concat=AGENT_MADDPGV2_GNN_CRITIC_BOOL_CONCAT,
        #                                          gnn_input_dims=AGENT_MADDPGV2_GNN_CRITIC_GNN_INPUT_DIMS,
        #                                          gnn_output_dims=AGENT_MADDPGV2_GNN_CRITIC_GNN_INPUT_DIMS,
        #                                          gmt_hidden_dims=AGENT_MADDPGV2_GNN_CRITIC_GMT_HIDDEN_DIMS,
        #                                          gmt_output_dims=AGENT_MADDPGV2_GNN_CRITIC_GMT_OUTPUT_DIMS,
        #                                          u_actions_fc_input_dims=AGENT_MADDPGV2_GNN_CRITIC_U_ACTIONS_FC_INPUT_DIMS,
        #                                          u_actions_fc_output_dims=AGENT_MADDPGV2_GNN_CRITIC_U_ACTIONS_FC_OUTPUT_DIMS,
        #                                          c_actions_fc_input_dims=AGENT_MADDPGV2_GNN_CRITIC_C_ACTIONS_FC_INPUT_DIMS,
        #                                          c_actions_fc_output_dims=AGENT_MADDPGV2_GNN_CRITIC_C_ACTIONS_FC_OUTPUT_DIMS,
        #                                          goal_fc_input_dims=AGENT_MADDPGV2_GNN_CRITIC_GOAL_FC_INPUT_DIMS,
        #                                          goal_fc_output_dims=AGENT_MADDPGV2_GNN_CRITIC_GOAL_FC_OUTPUT_DIMS,
        #                                          concat_fc_output_dims=AGENT_MADDPGV2_GNN_CRITIC_CONCAT_FC_OUTPUT_DIMS,
        #                                          tau=AGENT_MADDPGV2_GNN_TAU, mem_size=AGENT_MADDPGV2_GNN_MEMORY_SIZE,
        #                                          batch_size=AGENT_MADDPGV2_GNN_BATCH_SIZE,
        #                                          update_target=AGENT_MADDPGV2_GNN_UPDATE_TARGET,
        #                                          grad_clipping=AGENT_MADDPGV2_GNN_GRADIENT_CLIPPING,
        #                                          grad_norm_clip=AGENT_MADDPGV2_GNN_GRADIENT_NORM_CLIP,
        #                                          num_of_add_goals=AGENT_MADDPGV2_GNN_ADDITIONAL_GOALS,
        #                                          goal_strategy=AGENT_MADDPGV2_GNN_GOAL_STRATEGY, is_adversary=False,
        #                                          ep_time_limit=EPISODE_TIME_STEP_LIMIT, r_rad=RESTRICTED_RADIUS,
        #                                          big_rew_cnst=BIG_REWARD_CONSTANT,
        #                                          rew_multiplier_cnst=REWARD_MULTIPLIER_CONSTANT,
        #                                          pos_dims=POSITION_DIMENSIONS,
        #                                          exit_screen_terminate=EXIT_SCREEN_TERMINATE)

    # check adversarial model
    if ADVER_MODEL == "maddpg_gnn":
        pass
        # generate maddpg gnn agents for adversarial drones
        # adver_maddpg_gnn_agents = maddpg_gnn(mode=ADVER_MODE, scenario_name=SCENARIO_NAME,
        #                                      training_name=ADVER_TRAINING_NAME,
        #                                      discount_rate=ADVER_MADDPG_GNN_DISCOUNT_RATE,
        #                                      lr_actor=ADVER_MADDPG_GNN_LEARNING_RATE_ACTOR,
        #                                      lr_critic=ADVER_MADDPG_GNN_LEARNING_RATE_CRITIC,
        #                                      num_agents=NUMBER_OF_AGENT_DRONES,
        #                                      num_opp=NUMBER_OF_ADVER_DRONES,
        #                                      actor_dropout_p=ADVER_MADDPG_GNN_ACTOR_DROPOUT,
        #                                      critic_dropout_p=ADVER_MADDPG_GNN_CRITIC_DROPOUT,
        #                                      state_fc_input_dims=ADVER_MADDPG_GNN_ACTOR_INPUT_DIMENSIONS,
        #                                      state_fc_output_dims=ADVER_MADDPG_GNN_ACTOR_OUTPUT_DIMENSIONS,
        #                                      u_action_dims=ADVER_MADDPG_GNN_U_ACTIONS_DIMENSIONS,
        #                                      c_action_dims=ADVER_MADDPG_GNN_C_ACTIONS_DIMENSIONS,
        #                                      num_heads=ADVER_MADDPG_GNN_CRITIC_GNN_NUM_HEADS,
        #                                      bool_concat=ADVER_MADDPG_GNN_CRITIC_BOOL_CONCAT,
        #                                      gnn_input_dims=ADVER_MADDPG_GNN_CRITIC_GNN_INPUT_DIMS,
        #                                      gnn_output_dims=ADVER_MADDPG_GNN_CRITIC_GNN_INPUT_DIMS,
        #                                      gmt_hidden_dims=ADVER_MADDPG_GNN_CRITIC_GMT_HIDDEN_DIMS,
        #                                      gmt_output_dims=ADVER_MADDPG_GNN_CRITIC_GMT_OUTPUT_DIMS,
        #                                      u_actions_fc_input_dims=ADVER_MADDPG_GNN_CRITIC_U_ACTIONS_FC_INPUT_DIMS,
        #                                      u_actions_fc_output_dims=ADVER_MADDPG_GNN_CRITIC_U_ACTIONS_FC_OUTPUT_DIMS,
        #                                      c_actions_fc_input_dims=ADVER_MADDPG_GNN_CRITIC_C_ACTIONS_FC_INPUT_DIMS,
        #                                      c_actions_fc_output_dims=ADVER_MADDPG_GNN_CRITIC_C_ACTIONS_FC_OUTPUT_DIMS,
        #                                      concat_fc_output_dims=ADVER_MADDPG_GNN_CRITIC_CONCAT_FC_OUTPUT_DIMS,
        #                                      tau=ADVER_MADDPG_GNN_TAU, mem_size=ADVER_MADDPG_GNN_MEMORY_SIZE,
        #                                      batch_size=ADVER_MADDPG_GNN_BATCH_SIZE,
        #                                      update_target=ADVER_MADDPG_GNN_UPDATE_TARGET,
        #                                      grad_clipping=ADVER_MADDPG_GNN_GRADIENT_CLIPPING,
        #                                      grad_norm_clip=ADVER_MADDPG_GNN_GRADIENT_NORM_CLIP, is_adversary=True)

    elif ADVER_MODEL == "mappo_gnn":

        # generate mappo gnn agents for adver drones
        adver_mappo_gnn_agents = mappo_gnn(mode=AGENT_MODE, scenario_name=SCENARIO_NAME,
                                           training_name=AGENT_TRAINING_NAME,
                                           lr_actor=AGENT_MAPPO_GNN_LEARNING_RATE_ACTOR,
                                           lr_critic=AGENT_MAPPO_GNN_LEARNING_RATE_CRITIC,
                                           num_agents=NUMBER_OF_AGENT_DRONES, num_opp=NUMBER_OF_ADVER_DRONES,
                                           u_range=AGENT_DRONE_U_RANGE,
                                           u_noise=AGENT_DRONE_U_NOISE, c_noise=AGENT_DRONE_C_NOISE, is_adversary=False,
                                           actor_dropout_p=AGENT_MAPPO_GNN_ACTOR_DROPOUT,
                                           critic_dropout_p=AGENT_MAPPO_GNN_CRITIC_DROPOUT,
                                           state_fc_input_dims=AGENT_MAPPO_GNN_ACTOR_INPUT_DIMENSIONS,
                                           state_fc_output_dims=AGENT_MAPPO_GNN_ACTOR_OUTPUT_DIMENSIONS,
                                           u_action_dims=AGENT_MAPPO_GNN_U_ACTIONS_DIMENSIONS,
                                           c_action_dims=AGENT_MAPPO_GNN_C_ACTIONS_DIMENSIONS,
                                           num_heads=AGENT_MAPPO_GNN_CRITIC_GNN_NUM_HEADS,
                                           bool_concat=AGENT_MAPPO_GNN_CRITIC_BOOL_CONCAT,
                                           gnn_input_dims=AGENT_MAPPO_GNN_CRITIC_GNN_INPUT_DIMS,
                                           gnn_output_dims=AGENT_MAPPO_GNN_CRITIC_GNN_OUTPUT_DIMS,
                                           gmt_hidden_dims=AGENT_MAPPO_GNN_CRITIC_GMT_HIDDEN_DIMS,
                                           gmt_output_dims=AGENT_MAPPO_GNN_CRITIC_GMT_OUTPUT_DIMS,
                                           fc_output_dims=AGENT_MAPPO_GNN_CRITIC_FC_OUTPUT_DIMS,
                                           batch_size=AGENT_MAPPO_GNN_BATCH_SIZE, gamma=AGENT_MAPPO_GNN_GAMMA,
                                           clip_coeff=AGENT_MAPPO_GNN_CLIP_COEFFICIENT,
                                           num_epochs=AGENT_MAPPO_GNN_NUMBER_OF_EPOCHS,
                                           gae_lambda=AGENT_MAPPO_GNN_GAE_LAMBDA,
                                           entropy_coeff=AGENT_MAPPO_GNN_ENTROPY_COEFFICIENT,
                                           use_huber_loss=AGENT_MAPPO_GNN_USE_HUBER_LOSS,
                                           huber_delta=AGENT_MAPPO_GNN_HUBER_DELTA,
                                           use_clipped_value_loss=AGENT_MAPPO_GNN_USE_CLIPPED_VALUE_LOSS,
                                           critic_loss_coeff=AGENT_MAPPO_GNN_CRITIC_LOSS_COEFFICIENT,
                                           grad_clipping=AGENT_MAPPO_GNN_GRADIENT_CLIPPING,
                                           grad_norm_clip=AGENT_MAPPO_GNN_GRADIENT_NORM_CLIP,
                                           optimizer=OPTIMIZER, lr_scheduler=LR_SCHEDULER, obs_dims=OBS_DIMS,
                                           dgcn_output_dims=DGCN_OUTPUT_DIMS,
                                           somu_lstm_hidden_size=SOMU_LSTM_HIDDEN_SIZE,
                                           somu_lstm_num_layers=SOMU_LSTM_NUM_LAYERS,
                                           somu_lstm_dropout=SOMU_LSTM_DROPOUT, num_somu_lstm=NUM_SOMU_LSTM,
                                           scmu_lstm_hidden_size=SCMU_LSTM_HIDDEN_SIZE,
                                           scmu_lstm_num_layers=SCMU_LSTM_NUM_LAYERS,
                                           scmu_lstm_dropout=SCMU_LSTM_DROPOUT,
                                           num_scmu_lstm=NUM_SCMU_LSTM,
                                           somu_multi_att_num_heads=SOMU_MULTI_ATT_NUM_HEADS,
                                           somu_multi_att_dropout=SOMU_MULTI_ATT_DROPOUT,
                                           scmu_multi_att_num_heads=SCMU_MULTI_ATT_NUM_HEADS,
                                           scmu_multi_att_dropout=SCMU_MULTI_ATT_DROPOUT,
                                           actor_fc_output_dims=ACTOR_FC_OUTPUT_DIMS,
                                           actor_fc_dropout_p=ACTOR_FC_DROPOUT_P,
                                           softmax_actions_dims=SOFTMAX_ACTIONS_DIMS,
                                           softmax_actions_dropout_p=SOFTMAX_ACTIONS_DROPOUT_P
                                           )

    elif ADVER_MODEL == "maddpgv2_gnn":
        pass
        # generate maddpgv2 gnn agents for adver drones
        # adver_maddpgv2_gnn_agents = maddpgv2_gnn(mode=ADVER_MODE, scenario_name=SCENARIO_NAME,
        #                                          training_name=ADVER_TRAINING_NAME,
        #                                          discount_rate=ADVER_MADDPGV2_GNN_DISCOUNT_RATE,
        #                                          lr_actor=ADVER_MADDPGV2_GNN_LEARNING_RATE_ACTOR,
        #                                          lr_critic=ADVER_MADDPGV2_GNN_LEARNING_RATE_CRITIC,
        #                                          num_agents=NUMBER_OF_AGENT_DRONES,
        #                                          num_opp=NUMBER_OF_ADVER_DRONES,
        #                                          actor_dropout_p=ADVER_MADDPGV2_GNN_ACTOR_DROPOUT,
        #                                          critic_dropout_p=ADVER_MADDPGV2_GNN_CRITIC_DROPOUT,
        #                                          state_fc_input_dims=ADVER_MADDPGV2_GNN_ACTOR_INPUT_DIMENSIONS,
        #                                          state_fc_output_dims=ADVER_MADDPGV2_GNN_ACTOR_OUTPUT_DIMENSIONS,
        #                                          u_action_dims=ADVER_MADDPGV2_GNN_U_ACTIONS_DIMENSIONS,
        #                                          c_action_dims=ADVER_MADDPGV2_GNN_C_ACTIONS_DIMENSIONS,
        #                                          num_heads=ADVER_MADDPGV2_GNN_CRITIC_GNN_NUM_HEADS,
        #                                          bool_concat=ADVER_MADDPGV2_GNN_CRITIC_BOOL_CONCAT,
        #                                          gnn_input_dims=ADVER_MADDPGV2_GNN_CRITIC_GNN_INPUT_DIMS,
        #                                          gnn_output_dims=ADVER_MADDPGV2_GNN_CRITIC_GNN_INPUT_DIMS,
        #                                          gmt_hidden_dims=ADVER_MADDPGV2_GNN_CRITIC_GMT_HIDDEN_DIMS,
        #                                          gmt_output_dims=ADVER_MADDPGV2_GNN_CRITIC_GMT_OUTPUT_DIMS,
        #                                          u_actions_fc_input_dims=ADVER_MADDPGV2_GNN_CRITIC_U_ACTIONS_FC_INPUT_DIMS,
        #                                          u_actions_fc_output_dims=ADVER_MADDPGV2_GNN_CRITIC_U_ACTIONS_FC_OUTPUT_DIMS,
        #                                          c_actions_fc_input_dims=ADVER_MADDPGV2_GNN_CRITIC_C_ACTIONS_FC_INPUT_DIMS,
        #                                          c_actions_fc_output_dims=ADVER_MADDPGV2_GNN_CRITIC_C_ACTIONS_FC_OUTPUT_DIMS,
        #                                          goal_fc_input_dims=ADVER_MADDPGV2_GNN_CRITIC_GOAL_FC_INPUT_DIMS,
        #                                          goal_fc_output_dims=ADVER_MADDPGV2_GNN_CRITIC_GOAL_FC_OUTPUT_DIMS,
        #                                          concat_fc_output_dims=ADVER_MADDPGV2_GNN_CRITIC_CONCAT_FC_OUTPUT_DIMS,
        #                                          tau=ADVER_MADDPGV2_GNN_TAU, mem_size=ADVER_MADDPGV2_GNN_MEMORY_SIZE,
        #                                          batch_size=ADVER_MADDPGV2_GNN_BATCH_SIZE,
        #                                          update_target=ADVER_MADDPGV2_GNN_UPDATE_TARGET,
        #                                          grad_clipping=ADVER_MADDPGV2_GNN_GRADIENT_CLIPPING,
        #                                          grad_norm_clip=ADVER_MADDPGV2_GNN_GRADIENT_NORM_CLIP,
        #                                          num_of_add_goals=ADVER_MADDPGV2_GNN_ADDITIONAL_GOALS,
        #                                          goal_strategy=ADVER_MADDPGV2_GNN_GOAL_STRATEGY, is_adversary=True,
        #                                          ep_time_limit=EPISODE_TIME_STEP_LIMIT, r_rad=RESTRICTED_RADIUS,
        #                                          big_rew_cnst=BIG_REWARD_CONSTANT,
        #                                          rew_multiplier_cnst=REWARD_MULTIPLIER_CONSTANT,
        #                                          pos_dims=POSITION_DIMENSIONS,
        #                                          exit_screen_terminate=EXIT_SCREEN_TERMINATE)

    if AGENT_MODE != "test" and ADVER_MODE != "test":

        # generate environment during evaluation
        env = make_env(scenario_name=SCENARIO_NAME, dim_c=COMMUNICATION_DIMENSIONS,
                       num_good_agents=NUMBER_OF_AGENT_DRONES, num_adversaries=NUMBER_OF_ADVER_DRONES,
                       num_landmarks=NUMBER_OF_LANDMARKS, r_rad=RESTRICTED_RADIUS, i_rad=INTERCEPT_RADIUS,
                       r_noise_pos=RADAR_NOISE_POSITION, r_noise_vel=RADAR_NOISE_VELOCITY,
                       big_rew_cnst=BIG_REWARD_CONSTANT, rew_multiplier_cnst=REWARD_MULTIPLIER_CONSTANT,
                       ep_time_step_limit=EPISODE_TIME_STEP_LIMIT,
                       drone_radius=[AGENT_DRONE_RADIUS, ADVER_DRONE_RADIUS],
                       agent_size=[AGENT_DRONE_SIZE, ADVER_DRONE_SIZE],
                       agent_density=[AGENT_DRONE_DENSITY, ADVER_DRONE_DENSITY],
                       agent_initial_mass=[AGENT_DRONE_INITIAL_MASS, ADVER_DRONE_INITIAL_MASS],
                       agent_accel=[AGENT_DRONE_ACCEL, ADVER_DRONE_ACCEL],
                       agent_max_speed=[AGENT_DRONE_MAX_SPEED, ADVER_DRONE_MAX_SPEED],
                       agent_collide=[AGENT_DRONE_COLLIDE, ADVER_DRONE_COLLIDE],
                       agent_silent=[AGENT_DRONE_SILENT, ADVER_DRONE_SILENT],
                       agent_u_noise=[AGENT_DRONE_U_NOISE, ADVER_DRONE_U_NOISE],
                       agent_c_noise=[AGENT_DRONE_C_NOISE, ADVER_DRONE_C_NOISE],
                       agent_u_range=[AGENT_DRONE_U_RANGE, ADVER_DRONE_U_RANGE], landmark_size=LANDMARK_SIZE,
                       benchmark=True)

    elif AGENT_MODE == "test" and ADVER_MODE != "test":

        # generate environment during evaluation
        env = make_env(scenario_name=SCENARIO_NAME, dim_c=COMMUNICATION_DIMENSIONS,
                       num_good_agents=NUMBER_OF_AGENT_DRONES, num_adversaries=NUMBER_OF_ADVER_DRONES,
                       num_landmarks=NUMBER_OF_LANDMARKS, r_rad=RESTRICTED_RADIUS, i_rad=INTERCEPT_RADIUS,
                       r_noise_pos=RADAR_NOISE_POSITION, r_noise_vel=RADAR_NOISE_VELOCITY,
                       big_rew_cnst=BIG_REWARD_CONSTANT, rew_multiplier_cnst=REWARD_MULTIPLIER_CONSTANT,
                       ep_time_step_limit=EPISODE_TIME_STEP_LIMIT,
                       drone_radius=[AGENT_DRONE_RADIUS, ADVER_DRONE_RADIUS],
                       agent_size=[AGENT_DRONE_SIZE, ADVER_DRONE_SIZE],
                       agent_density=[AGENT_DRONE_DENSITY, ADVER_DRONE_DENSITY],
                       agent_initial_mass=[AGENT_DRONE_INITIAL_MASS, ADVER_DRONE_INITIAL_MASS],
                       agent_accel=[AGENT_DRONE_ACCEL, ADVER_DRONE_ACCEL],
                       agent_max_speed=[AGENT_DRONE_MAX_SPEED, ADVER_DRONE_MAX_SPEED],
                       agent_collide=[AGENT_DRONE_COLLIDE, ADVER_DRONE_COLLIDE],
                       agent_silent=[AGENT_DRONE_SILENT, ADVER_DRONE_SILENT], agent_u_noise=[0.0, ADVER_DRONE_U_NOISE],
                       agent_c_noise=[0.0, ADVER_DRONE_C_NOISE],
                       agent_u_range=[AGENT_DRONE_U_RANGE, ADVER_DRONE_U_RANGE], landmark_size=LANDMARK_SIZE,
                       benchmark=True)

    elif AGENT_MODE != "test" and ADVER_MODE == "test":

        # generate environment during evaluation
        env = make_env(scenario_name=SCENARIO_NAME, dim_c=COMMUNICATION_DIMENSIONS,
                       num_good_agents=NUMBER_OF_AGENT_DRONES, num_adversaries=NUMBER_OF_ADVER_DRONES,
                       num_landmarks=NUMBER_OF_LANDMARKS, r_rad=RESTRICTED_RADIUS, i_rad=INTERCEPT_RADIUS,
                       r_noise_pos=RADAR_NOISE_POSITION, r_noise_vel=RADAR_NOISE_VELOCITY,
                       big_rew_cnst=BIG_REWARD_CONSTANT, rew_multiplier_cnst=REWARD_MULTIPLIER_CONSTANT,
                       ep_time_step_limit=EPISODE_TIME_STEP_LIMIT,
                       drone_radius=[AGENT_DRONE_RADIUS, ADVER_DRONE_RADIUS],
                       agent_size=[AGENT_DRONE_SIZE, ADVER_DRONE_SIZE],
                       agent_density=[AGENT_DRONE_DENSITY, ADVER_DRONE_DENSITY],
                       agent_initial_mass=[AGENT_DRONE_INITIAL_MASS, ADVER_DRONE_INITIAL_MASS],
                       agent_accel=[AGENT_DRONE_ACCEL, ADVER_DRONE_ACCEL],
                       agent_max_speed=[AGENT_DRONE_MAX_SPEED, ADVER_DRONE_MAX_SPEED],
                       agent_collide=[AGENT_DRONE_COLLIDE, ADVER_DRONE_COLLIDE],
                       agent_silent=[AGENT_DRONE_SILENT, ADVER_DRONE_SILENT], agent_u_noise=[AGENT_DRONE_U_NOISE, 0.0],
                       agent_c_noise=[AGENT_DRONE_C_NOISE, 0.0],
                       agent_u_range=[AGENT_DRONE_U_RANGE, ADVER_DRONE_U_RANGE], landmark_size=LANDMARK_SIZE,
                       benchmark=True)

    elif AGENT_MODE == "test" and ADVER_MODE == "test":

        # generate environment during evaluation
        env = make_env(scenario_name=SCENARIO_NAME, dim_c=COMMUNICATION_DIMENSIONS,
                       num_good_agents=NUMBER_OF_AGENT_DRONES, num_adversaries=NUMBER_OF_ADVER_DRONES,
                       num_landmarks=NUMBER_OF_LANDMARKS, r_rad=RESTRICTED_RADIUS, i_rad=INTERCEPT_RADIUS,
                       r_noise_pos=RADAR_NOISE_POSITION, r_noise_vel=RADAR_NOISE_VELOCITY,
                       big_rew_cnst=BIG_REWARD_CONSTANT, rew_multiplier_cnst=REWARD_MULTIPLIER_CONSTANT,
                       ep_time_step_limit=EPISODE_TIME_STEP_LIMIT,
                       drone_radius=[AGENT_DRONE_RADIUS, ADVER_DRONE_RADIUS],
                       agent_size=[AGENT_DRONE_SIZE, ADVER_DRONE_SIZE],
                       agent_density=[AGENT_DRONE_DENSITY, ADVER_DRONE_DENSITY],
                       agent_initial_mass=[AGENT_DRONE_INITIAL_MASS, ADVER_DRONE_INITIAL_MASS],
                       agent_accel=[AGENT_DRONE_ACCEL, ADVER_DRONE_ACCEL],
                       agent_max_speed=[AGENT_DRONE_MAX_SPEED, ADVER_DRONE_MAX_SPEED],
                       agent_collide=[AGENT_DRONE_COLLIDE, ADVER_DRONE_COLLIDE],
                       agent_silent=[AGENT_DRONE_SILENT, ADVER_DRONE_SILENT], agent_u_noise=[0.0, 0.0],
                       agent_c_noise=[0.0, 0.0], agent_u_range=[AGENT_DRONE_U_RANGE, ADVER_DRONE_U_RANGE],
                       landmark_size=LANDMARK_SIZE, benchmark=True)

    # if log directory for tensorboard exist
    if os.path.exists(TENSORBOARD_LOG_DIRECTORY):
        # remove entire directory
        shutil.rmtree(TENSORBOARD_LOG_DIRECTORY)

    # generate writer for tensorboard logging
    writer = SummaryWriter(log_dir=TENSORBOARD_LOG_DIRECTORY)

    # generate edge_index for complete graph for gnn for critic models
    agent_edge_index = complete_graph_edge_index(num_nodes=NUMBER_OF_AGENT_DRONES)
    adver_edge_index = complete_graph_edge_index(num_nodes=NUMBER_OF_ADVER_DRONES)

    # variables to track elo of agent and adver models
    agent_elo = INITIAL_ELO
    adver_elo = INITIAL_ELO

    # variables to track the sum of agent and opp wins
    sum_agent_wins = 0
    sum_adver_wins = 0

    # variables to track for exiting screen
    sum_agent_exceed_screen = 0
    sum_adver_exceed_screen = 0

    # list to store metrics to be converted to csv for postprocessing
    sum_agent_wins_list = []
    sum_adver_wins_list = []
    sum_agent_number_of_team_collisions_list = []
    sum_agent_number_of_oppo_collisions_list = []
    sum_adver_number_of_team_collisions_list = []
    sum_adver_number_of_oppo_collisions_list = []
    sum_agent_exceed_screen_list = []
    sum_adver_exceed_screen_list = []
    sum_adver_disabled_list = []

    avg_agent_actor_loss_list = []
    avg_agent_critic_loss_list = []
    avg_adver_actor_loss_list = []
    avg_adver_critic_loss_list = []
    avg_agent_number_of_team_collisions_list = []
    avg_agent_number_of_oppo_collisions_list = []
    avg_adver_number_of_team_collisions_list = []
    avg_adver_number_of_oppo_collisions_list = []
    avg_agent_actor_grad_norm_list = []
    avg_agent_critic_grad_norm_list = []
    avg_adver_actor_grad_norm_list = []
    avg_adver_critic_grad_norm_list = []
    avg_agent_policy_ratio_list = []
    avg_adver_policy_ratio_list = []

    # list to store agent and adver elo
    agent_elo_list = []
    adver_elo_list = []

    # list to store goals
    agent_goals_list = []
    adver_goals_list = []

    # check if agent model is mappo_gnn
    if AGENT_MODEL == "mappo_gnn":
        # generate batch tensor for graph multiset transformer in critic model for agent for mappo_gnn
        agent_critic_batch = T.tensor([i for i in range(1) for j in range(NUMBER_OF_AGENT_DRONES)], dtype=T.long).to(
            T.device('cuda:0' if T.cuda.is_available() else 'cpu'))

        # variable to track number of steps in episode
        agent_eps_steps = 0

    # check if adver model is mappo_gnn
    if ADVER_MODEL == "mappo_gnn":
        # generate batch tensor for graph multiset transformer in critic model for adver for mappo_gnn
        adver_critic_batch = T.tensor([i for i in range(1) for j in range(NUMBER_OF_ADVER_DRONES)], dtype=T.long).to(
            T.device('cuda:0' if T.cuda.is_available() else 'cpu'))

        # variable to track number of steps in episode
        adver_eps_steps = 0

    # goals based variables for agent maddppv2_gnn
    if AGENT_MODEL == "maddpgv2_gnn":
        # initialise softmax weights for agent goal distribution
        agent_goals_softmax_weights = np.zeros(len(AGENT_MADDPGV2_GNN_GOAL_DISTRIBUTION), dtype=np.float64)

    # goals based variables for agent maddppv2_gnn
    if ADVER_MODEL == "maddpgv2_gnn":
        # initialise softmax weights for adversarial goal distribution
        adver_goals_softmax_weights = np.zeros(len(ADVER_MADDPGV2_GNN_GOAL_DISTRIBUTION), dtype=np.float64)

    # iterate over number of episodes
    for eps in range(1, NUMBER_OF_EPISODES + 1):

        # boolean to check if episode is terminal
        is_terminal = 0

        # variable to track terminal condition
        terminal_condition = 0

        # print episode number
        print("episode " + str(eps) + ":")

        # obtain states of agent and adverserial agents
        actor_states = env.reset()

        # check if exponential decay for noise is desired
        if EXPONENTIAL_NOISE_DECAY == True:
            update_noise_exponential_decay(env=env, expo_decay_cnst=EXPONENTIAL_NOISE_DECAY_CONSTANT,
                                           num_adver=NUMBER_OF_ADVER_DRONES, eps_timestep=eps,
                                           agent_u_noise_cnst=AGENT_DRONE_U_NOISE,
                                           agent_c_noise_cnst=AGENT_DRONE_C_NOISE,
                                           adver_u_noise_cnst=ADVER_DRONE_U_NOISE,
                                           adver_c_noise_cnst=ADVER_DRONE_C_NOISE)

        # set up agent actor goals for maddpg_gnn and mappo_gnn
        if AGENT_MODEL == "maddpg_gnn" or AGENT_MODEL == "mappo_gnn":

            agent_goal = AGENT_MADDPGV2_GNN_GOAL
            agent_actor_goals = np.array([[AGENT_MADDPGV2_GNN_GOAL] for i in range(NUMBER_OF_AGENT_DRONES)])

        # set up agent actor and critic goals for maddpgv2_gnn
        elif AGENT_MODEL == "maddpgv2_gnn":

            # check if training
            if AGENT_MODE != "test":

                # obtain probability distribution from agent_goals_softmax_weights
                prob_dist = softmax(agent_goals_softmax_weights)

                # sample goal from probability distribution
                agent_goal = np.random.choice(a=AGENT_MADDPGV2_GNN_GOAL_DISTRIBUTION, p=prob_dist)

                # obtain agent_actor_goals and agent_critic_goals
                agent_actor_goals = np.array([[agent_goal] for i in range(NUMBER_OF_AGENT_DRONES)])
                agent_critic_goals = np.array([[agent_goal] for i in range(NUMBER_OF_AGENT_DRONES)]).reshape(1, -1)

            else:

                # set for original goals
                agent_goal = AGENT_MADDPGV2_GNN_GOAL
                agent_actor_goals = np.array([[agent_goal] for i in range(NUMBER_OF_AGENT_DRONES)])
                agent_critic_goals = np.array([[agent_goal] for i in range(NUMBER_OF_AGENT_DRONES)]).reshape(1, -1)

        # set up adver actor goals for maddpg_gnn and mappo_gnn
        if ADVER_MODEL == "maddpg_gnn" or ADVER_MODEL == "mappo_gnn":
            adver_goal = ADVER_MADDPGV2_GNN_GOAL
            adver_actor_goals = np.array([[ADVER_MADDPGV2_GNN_GOAL] for i in range(NUMBER_OF_ADVER_DRONES)])

        # set up adver actor and critic goals for maddpgv2_gnn
        if ADVER_MODEL == "maddpgv2_gnn":

            # check if training
            if ADVER_MODE != "test":

                # obtain probability distribution from adver_goals_softmax_weights
                prob_dist = softmax(adver_goals_softmax_weights)

                # sample goal from probability distribution
                adver_goal = np.random.choice(a=ADVER_MADDPGV2_GNN_GOAL_DISTRIBUTION, p=prob_dist)

                # obtain adver_actor_goals and adver_critic_goals
                adver_actor_goals = np.array([[adver_goal] for i in range(NUMBER_OF_ADVER_DRONES)])
                adver_critic_goals = np.array([[adver_goal] for i in range(NUMBER_OF_ADVER_DRONES)]).reshape(1, -1)

            else:

                # set for original goals
                adver_goal = ADVER_MADDPGV2_GNN_GOAL
                adver_actor_goals = np.array([[adver_goal] for i in range(NUMBER_OF_ADVER_DRONES)])
                adver_critic_goals = np.array([[adver_goal] for i in range(NUMBER_OF_ADVER_DRONES)]).reshape(1, -1)

        # obtain numpy array of actor_states, adver_actor_states, agent_actor_states
        actor_states = np.array(actor_states)
        adver_actor_states = np.array(actor_states[:NUMBER_OF_ADVER_DRONES])
        agent_actor_states = np.array(actor_states[NUMBER_OF_ADVER_DRONES:])

        # variables to track metrics for agent and adver
        sum_agent_actor_loss = 0
        sum_agent_critic_loss = 0
        sum_agent_number_of_team_collisions = 0
        sum_agent_number_of_oppo_collisions = 0
        sum_agent_actor_grad_norm = 0
        sum_agent_critic_grad_norm = 0
        sum_agent_policy_ratio = 0
        sum_adver_actor_loss = 0
        sum_adver_critic_loss = 0
        sum_adver_number_of_team_collisions = 0
        sum_adver_number_of_oppo_collisions = 0
        sum_adver_actor_grad_norm = 0
        sum_adver_critic_grad_norm = 0
        sum_adver_policy_ratio = 0
        sum_adver_disabled = 0

        # iterate till episode terminates
        while is_terminal == 0:

            # check if environment is required to be rendered
            if RENDER_ENV == True:
                # render env
                env.render()

            # obtain actions for agent_maddpg_gnn_agents
            # if AGENT_MODEL == "maddpg_gnn":
            #
            #     # obtain motor and communication actions for agent drones
            #     # mode is always 'test' as the environment handles the addition of noise to the actions
            #     agent_u_actions, agent_c_actions, agent_actions_list = agent_maddpg_gnn_agents.select_actions(
            #         mode="test", env_agents=env.agents[NUMBER_OF_ADVER_DRONES:],
            #         actor_state_list=agent_actor_states)

            # obtain actions for agent_mappo_gnn_agents
            if AGENT_MODEL == "mappo_gnn":
                # obtain motor and communication actions for agent drones
                # mode is always 'test' as the environment handles the addition of noise to the actions
                agent_u_actions, agent_c_actions, agent_u_actions_log_probs, agent_c_actions_log_probs, agent_actions_list = \
                    agent_mappo_gnn_agents.select_actions(mode="test", env_agents=env.agents[NUMBER_OF_ADVER_DRONES:],
                                                          actor_state_list=agent_actor_states)

            # obtain actions for agent_maddpgv2_gnn_agents
            # elif AGENT_MODEL == "maddpgv2_gnn":
            #
            #     # agent_actor_states concatenated with agent goals
            #     agent_actor_states_p_goal = np.concatenate((agent_actor_states, agent_actor_goals), axis=-1)
            #
            #     # obtain motor and communication actions for agent drones
            #     # mode is always 'test' as the environment handles the addition of noise to the actions
            #     agent_u_actions, agent_c_actions, agent_actions_list = agent_maddpgv2_gnn_agents.select_actions(
            #         mode="test", env_agents=env.agents[NUMBER_OF_ADVER_DRONES:],
            #         actor_state_list=agent_actor_states_p_goal)

            # obtain actions for adver_maddpg_gnn_agents
            # if ADVER_MODEL == "maddpg_gnn":
            #
            #     # obtain actions from fc_state and cam_state for all opp drones
            #     # mode is always 'test' as the environment handles the addition of noise to the actions
            #     adver_u_actions, adver_c_actions, adver_actions_list = adver_maddpg_gnn_agents.select_actions(
            #         mode="test", env_agents=env.agents[:NUMBER_OF_ADVER_DRONES],
            #         actor_state_list=adver_actor_states)

            # obtain actions for adver_mappo_gnn_agents
            if AGENT_MODEL == "mappo_gnn":
                # obtain motor and communication actions for adver drones
                # mode is always 'test' as the environment handles the addition of noise to the actions
                adver_u_actions, adver_c_actions, adver_u_actions_log_probs, adver_c_actions_log_probs, adver_actions_list = \
                    adver_mappo_gnn_agents.select_actions(mode="test", env_agents=env.agents[:NUMBER_OF_ADVER_DRONES],
                                                          actor_state_list=adver_actor_states)

            # obtain actions for adver_maddpgv2_gnn_agents
            # elif ADVER_MODEL == "maddpgv2_gnn":
            #
            #     # agent_actor_states concatenated with agent goals
            #     adver_actor_states_p_goal = np.concatenate((adver_actor_states, adver_actor_goals), axis=-1)
            #
            #     # obtain actions from fc_state and cam_state for all opp drones
            #     # mode is always 'test' as the environment handles the addition of noise to the actions
            #     adver_u_actions, adver_c_actions, adver_actions_list = adver_maddpgv2_gnn_agents.select_actions(
            #         mode="test", env_agents=env.agents[:NUMBER_OF_ADVER_DRONES],
            #         actor_state_list=adver_actor_states_p_goal)

            # iterate over agent drones
            for i in range(NUMBER_OF_AGENT_DRONES):
                # append agent drones actions to adversarial drones actions
                adver_actions_list.append(agent_actions_list[i])

            # update state of the world and obtain information of the updated state
            actor_states_prime, rewards, terminates_p_terminal_con, benchmark_data = env.step(
                action_n=adver_actions_list, agent_goal=AGENT_MADDPGV2_GNN_GOAL,
                adver_goal=ADVER_MADDPGV2_GNN_GOAL)

            # update world ep_time_step
            env.world.ep_time_step += 1

            # obtain numpy array of actor_states_prime, adver_actor_states_prime, agent_actor_states_prime, adver_rewards, agent_rewards
            actor_states_prime = np.array(actor_states_prime)
            adver_actor_states_prime = np.array(actor_states_prime[:NUMBER_OF_ADVER_DRONES])
            agent_actor_states_prime = np.array(actor_states_prime[NUMBER_OF_ADVER_DRONES:])
            adver_rewards = np.array(rewards[:NUMBER_OF_ADVER_DRONES])
            agent_rewards = np.array(rewards[NUMBER_OF_ADVER_DRONES:])
            adver_benchmark_data = np.array(benchmark_data['n'][:NUMBER_OF_ADVER_DRONES])
            agent_benchmark_data = np.array(benchmark_data['n'][NUMBER_OF_ADVER_DRONES:])

            # empty list for adver_terminates, agent_terminates, terminal_con
            adver_terminates = []
            agent_terminates = []
            terminal_con = []

            # iterate over all drones
            for i in range(NUMBER_OF_AGENT_DRONES + NUMBER_OF_ADVER_DRONES):

                # check for adversarial drones
                if i < NUMBER_OF_ADVER_DRONES:

                    # append terminates and terminal_con
                    adver_terminates.append(terminates_p_terminal_con[i][0])
                    terminal_con.append(terminates_p_terminal_con[i][1])

                    # check if episode has terminated and is_terminal is false
                    if adver_terminates[i] == True and is_terminal == False:
                        # update is_terminal
                        is_terminal = True

                        # obtain corresponding terminal condition
                        terminal_condition = terminal_con[i]

                # check for agent drones
                if i >= NUMBER_OF_ADVER_DRONES:

                    # append terminates and terminal_con
                    agent_terminates.append(terminates_p_terminal_con[i][0])
                    terminal_con.append(terminates_p_terminal_con[i][1])

                    # check if episode has terminated and is_terminal is false
                    if agent_terminates[i - NUMBER_OF_ADVER_DRONES] == True and is_terminal == False:
                        # update is_terminal
                        is_terminal = True

                        # obtain corresponding terminal condition
                        terminal_condition = terminal_con[i]

            # obtain numpy array of adver_terminates, agent_terminates, terminal_con
            adver_terminates = np.array(adver_terminates, dtype=bool)
            agent_terminates = np.array(agent_terminates, dtype=bool)
            terminal_con = np.array(terminal_con)

            # for maddpg_gnn agent drones to store memory in replay buffer
            # if AGENT_MODEL == "maddpg_gnn":
            #
            #     # check if agent is training
            #     if AGENT_MODE != "test":
            #
            #         # obtain agent_critic_states and agent_critic_states_prime in gnn data format
            #         agent_critic_states = Data(x=T.tensor(agent_actor_states, dtype=T.float),
            #                                    edge_index=T.tensor(agent_edge_index, dtype=T.long).t().contiguous())
            #         agent_critic_states_prime = Data(x=T.tensor(agent_actor_states_prime, dtype=T.float),
            #                                          edge_index=T.tensor(agent_edge_index,
            #                                                              dtype=T.long).t().contiguous())
            #
            #         # set num_nodes for agent_critic_states, agent_critic_states_prime
            #         agent_critic_states.num_nodes = NUMBER_OF_AGENT_DRONES
            #         agent_critic_states_prime.num_nodes = NUMBER_OF_AGENT_DRONES
            #
            #         # store states and actions in replay buffer
            #         agent_maddpg_gnn_agents.replay_buffer.log(actor_state=agent_actor_states,
            #                                                   actor_state_prime=agent_actor_states_prime,
            #                                                   critic_state=agent_critic_states,
            #                                                   critic_state_prime=agent_critic_states_prime,
            #                                                   u_action=agent_u_actions, c_action=agent_c_actions,
            #                                                   rewards=agent_rewards,
            #                                                   is_done=agent_terminates)
            #
            #         # train model
            #         if not (abs(agent_elo - adver_elo) > ELO_DIFFRENCE and agent_elo > adver_elo):
            #
            #             # train agent models and obtain metrics for each agent drone for logging
            #             agent_actor_loss_list, agent_critic_loss_list, agent_actor_grad_norm_list, agent_critic_grad_norm_list = \
            #                 agent_maddpg_gnn_agents.apply_gradients_maddpg_gnn(num_of_agents=NUMBER_OF_AGENT_DRONES)
            #
            #         else:
            #
            #             agent_actor_loss_list, agent_critic_loss_list, agent_actor_grad_norm_list, agent_critic_grad_norm_list = np.nan, np.nan, np.nan, np.nan
            #
            #     else:
            #
            #         agent_actor_loss_list, agent_critic_loss_list, agent_actor_grad_norm_list, agent_critic_grad_norm_list = np.nan, np.nan, np.nan, np.nan

            # for mappo_gnn agent to store memory in replay buffer and train model
            if AGENT_MODEL == "mappo_gnn":

                # train model
                if AGENT_MODE != "test" and not (abs(agent_elo - adver_elo) > ELO_DIFFRENCE and agent_elo > adver_elo):

                    # obtain agent_critic_states in gnn data format
                    agent_critic_states = Data(x=T.tensor(agent_actor_states, dtype=T.float), edge_index= \
                        T.tensor(agent_edge_index, dtype=T.long).t().contiguous()).to(
                        T.device('cuda:0' if T.cuda.is_available() else 'cpu'))

                    # set num_nodes for agent_critic_states
                    agent_critic_states.num_nodes = NUMBER_OF_AGENT_DRONES

                    # list to store critic state values
                    agent_critic_state_value = []

                    # iterate over agent critic models to obtain agent_critic_state_value:
                    for agent_index, agent in enumerate(agent_mappo_gnn_agents.mappo_gnn_agents_list):
                        # turn critic to eval mode
                        agent.mappo_gnn_critic.eval()

                        # append critic value to list
                        agent_critic_state_value.append(
                            agent.mappo_gnn_critic.forward(agent_critic_states, agent_critic_batch).item())

                        # turn critic to train mode
                        agent.mappo_gnn_critic.train()

                    # obtain numpy array of agent_critic_state_value
                    agent_critic_state_value = np.array(agent_critic_state_value)

                    # obtain cpu copy of critic states
                    agent_critic_states = agent_critic_states.cpu()

                    # store states and actions in replay buffer
                    agent_mappo_gnn_agents.replay_buffer.log(actor_state=agent_actor_states,
                                                             critic_state=agent_critic_states,
                                                             critic_state_value=agent_critic_state_value,
                                                             u_action=agent_u_actions, c_action=agent_c_actions,
                                                             u_action_log_probs=agent_u_actions_log_probs,
                                                             c_action_log_probs=agent_c_actions_log_probs,
                                                             rewards=agent_rewards, is_done=agent_terminates)

                    # update agent_eps_steps
                    agent_eps_steps += 1

                    # train model
                    if agent_eps_steps % AGENT_MAPPO_GNN_EPISODE_LENGTH == 0:

                        # train agent models and obtain metrics for each agent drone for logging
                        agent_actor_loss_list, agent_critic_loss_list, agent_actor_grad_norm_list, agent_critic_grad_norm_list, agent_policy_ratio_list = \
                            agent_mappo_gnn_agents.apply_gradients_mappo_gnn(num_of_agents=NUMBER_OF_AGENT_DRONES)

                    else:

                        agent_actor_loss_list, agent_critic_loss_list, agent_actor_grad_norm_list, agent_critic_grad_norm_list, agent_policy_ratio_list = np.nan, np.nan, np.nan, np.nan, np.nan

                else:

                    agent_actor_loss_list, agent_critic_loss_list, agent_actor_grad_norm_list, agent_critic_grad_norm_list, agent_policy_ratio_list = np.nan, np.nan, np.nan, np.nan, np.nan

            # for maddpgv2_gnn agent drones to store memory in replay buffer
            # elif AGENT_MODEL == "maddpgv2_gnn":
            #
            #     # check if agent is training
            #     if AGENT_MODE != "test":
            #
            #         # obtain agent_critic_states and agent_critic_states_prime in gnn data format
            #         agent_critic_states = Data(x=T.tensor(agent_actor_states, dtype=T.float),
            #                                    edge_index=T.tensor(agent_edge_index, dtype=T.long).t().contiguous())
            #         agent_critic_states_prime = Data(x=T.tensor(agent_actor_states_prime, dtype=T.float),
            #                                          edge_index=T.tensor(agent_edge_index,
            #                                                              dtype=T.long).t().contiguous())
            #
            #         # set num_nodes for agent_critic_states, agent_critic_states_prime
            #         agent_critic_states.num_nodes = NUMBER_OF_AGENT_DRONES
            #         agent_critic_states_prime.num_nodes = NUMBER_OF_AGENT_DRONES
            #
            #         # store states and actions in replay buffer
            #         agent_maddpgv2_gnn_agents.replay_buffer.log(actor_state=agent_actor_states,
            #                                                     actor_state_prime=agent_actor_states_prime,
            #                                                     org_actor_goals=agent_actor_goals,
            #                                                     critic_state=agent_critic_states,
            #                                                     critic_state_prime=agent_critic_states_prime,
            #                                                     org_critic_goals=agent_critic_goals,
            #                                                     u_action=agent_u_actions, c_action=agent_c_actions,
            #                                                     org_rewards=agent_rewards, is_done=agent_terminates)
            #
            #         # train model
            #         if not (abs(agent_elo - adver_elo) > ELO_DIFFRENCE and agent_elo > adver_elo):
            #
            #             # train agent models and obtain metrics for each agent drone for logging
            #             agent_actor_loss_list, agent_critic_loss_list, agent_actor_grad_norm_list, agent_critic_grad_norm_list = \
            #                 agent_maddpgv2_gnn_agents.apply_gradients_maddpgv2_gnn(num_of_agents=NUMBER_OF_AGENT_DRONES)
            #
            #         else:
            #
            #             agent_actor_loss_list, agent_critic_loss_list, agent_actor_grad_norm_list, agent_critic_grad_norm_list = np.nan, np.nan, np.nan, np.nan
            #
            #     else:
            #
            #         agent_actor_loss_list, agent_critic_loss_list, agent_actor_grad_norm_list, agent_critic_grad_norm_list = np.nan, np.nan, np.nan, np.nan

            # for maddpg_gnn adversarial drones to store memory in replay buffer
            # if ADVER_MODEL == "maddpg_gnn":
            #
            #     # check if adversarial is training
            #     if ADVER_MODE != "test":
            #
            #         # obtain adver_critic_states and adver_critic_states_prime in gnn data format
            #         adver_critic_states = Data(x=T.tensor(adver_actor_states, dtype=T.float),
            #                                    edge_index=T.tensor(adver_edge_index, dtype=T.long).t().contiguous())
            #         adver_critic_states_prime = Data(x=T.tensor(adver_actor_states_prime, dtype=T.float),
            #                                          edge_index=T.tensor(adver_edge_index,
            #                                                              dtype=T.long).t().contiguous())
            #
            #         # set num_nodes for adver_critic_states, adver_critic_states_prime
            #         adver_critic_states.num_nodes = NUMBER_OF_ADVER_DRONES
            #         adver_critic_states_prime.num_nodes = NUMBER_OF_ADVER_DRONES
            #
            #         # store states and actions in replay buffer
            #         adver_maddpg_gnn_agents.replay_buffer.log(actor_state=adver_actor_states,
            #                                                   actor_state_prime=adver_actor_states_prime,
            #                                                   critic_state=adver_critic_states,
            #                                                   critic_state_prime=adver_critic_states_prime,
            #                                                   u_action=adver_u_actions, c_action=adver_c_actions,
            #                                                   rewards=adver_rewards,
            #                                                   is_done=adver_terminates)
            #
            #         # train model
            #         if not (abs(adver_elo - agent_elo) > ELO_DIFFRENCE and adver_elo > agent_elo):
            #
            #             # train adversarial models and obtain metrics for each adversarial drone for logging
            #             adver_actor_loss_list, adver_critic_loss_list, adver_actor_grad_norm_list, adver_critic_grad_norm_list = \
            #                 adver_maddpg_gnn_agents.apply_gradients_maddpg_gnn(num_of_agents=NUMBER_OF_ADVER_DRONES)
            #
            #         else:
            #
            #             adver_actor_loss_list, adver_critic_loss_list, adver_actor_grad_norm_list, adver_critic_grad_norm_list = np.nan, np.nan, np.nan, np.nan
            #
            #     else:
            #
            #         adver_actor_loss_list, adver_critic_loss_list, adver_actor_grad_norm_list, adver_critic_grad_norm_list = np.nan, np.nan, np.nan, np.nan

            # for mappo_gnn agent to store memory in replay buffer and train model
            if ADVER_MODEL == "mappo_gnn":

                # train model
                if ADVER_MODE != "test" and not (abs(adver_elo - agent_elo) > ELO_DIFFRENCE and adver_elo > agent_elo):

                    # obtain adver_critic_states in gnn data format
                    adver_critic_states = Data(x=T.tensor(adver_actor_states, dtype=T.float),
                                               edge_index=T.tensor(adver_edge_index, dtype= \
                                                   T.long).t().contiguous()).to(
                        T.device('cuda:0' if T.cuda.is_available() else 'cpu'))

                    # set num_nodes for adver_critic_states
                    adver_critic_states.num_nodes = NUMBER_OF_ADVER_DRONES

                    # list to store critic state values
                    adver_critic_state_value = []

                    # iterate over agent critic models to obtain agent_critic_state_value:
                    for agent_index, agent in enumerate(adver_mappo_gnn_agents.mappo_gnn_agents_list):
                        # turn critic to eval mode
                        agent.mappo_gnn_critic.eval()

                        # append critic value to list
                        adver_critic_state_value.append(
                            agent.mappo_gnn_critic.forward(adver_critic_states, adver_critic_batch).item())

                        # turn critic to train mode
                        agent.mappo_gnn_critic.train()

                    # obtain numpy array of adver_critic_state_value
                    adver_critic_state_value = np.array(adver_critic_state_value)

                    # obtain cpu copy of critic states
                    adver_critic_states = adver_critic_states.cpu()

                    # store states and actions in replay buffer
                    adver_mappo_gnn_agents.replay_buffer.log(actor_state=adver_actor_states,
                                                             critic_state=adver_critic_states,
                                                             critic_state_value=adver_critic_state_value,
                                                             u_action=adver_u_actions, c_action=adver_c_actions,
                                                             u_action_log_probs=adver_u_actions_log_probs,
                                                             c_action_log_probs=adver_c_actions_log_probs,
                                                             rewards=adver_rewards, is_done=adver_terminates)

                    # update adver_eps_steps
                    adver_eps_steps += 1

                    # train model
                    if adver_eps_steps % ADVER_MAPPO_GNN_EPISODE_LENGTH == 0:

                        # train agent models and obtain metrics for each adver drone for logging
                        adver_actor_loss_list, adver_critic_loss_list, adver_actor_grad_norm_list, adver_critic_grad_norm_list, adver_policy_ratio_list = \
                            adver_mappo_gnn_agents.apply_gradients_mappo_gnn(num_of_agents=NUMBER_OF_ADVER_DRONES)

                    else:

                        adver_actor_loss_list, adver_critic_loss_list, adver_actor_grad_norm_list, adver_critic_grad_norm_list, adver_policy_ratio_list = np.nan, np.nan, np.nan, np.nan, np.nan

                else:

                    adver_actor_loss_list, adver_critic_loss_list, adver_actor_grad_norm_list, adver_critic_grad_norm_list, adver_policy_ratio_list = np.nan, np.nan, np.nan, np.nan, np.nan

            # for maddpgv2_gnn adversarial drones to store memory in replay buffer
            # elif ADVER_MODEL == "maddpgv2_gnn":
            #
            #     # check if adversarial is training
            #     if ADVER_MODE != "test":
            #
            #         # obtain adver_critic_states and adver_critic_states_prime in gnn data format
            #         adver_critic_states = Data(x=T.tensor(adver_actor_states, dtype=T.float),
            #                                    edge_index=T.tensor(adver_edge_index, dtype=T.long).t().contiguous())
            #         adver_critic_states_prime = Data(x=T.tensor(adver_actor_states_prime, dtype=T.float),
            #                                          edge_index=T.tensor(adver_edge_index,
            #                                                              dtype=T.long).t().contiguous())
            #
            #         # set num_nodes for adver_critic_states, adver_critic_states_prime
            #         adver_critic_states.num_nodes = NUMBER_OF_ADVER_DRONES
            #         adver_critic_states_prime.num_nodes = NUMBER_OF_ADVER_DRONES
            #
            #         # store states and actions in replay buffer
            #         adver_maddpgv2_gnn_agents.replay_buffer.log(actor_state=adver_actor_states,
            #                                                     actor_state_prime=adver_actor_states_prime,
            #                                                     org_actor_goals=adver_actor_goals,
            #                                                     critic_state=adver_critic_states,
            #                                                     critic_state_prime=adver_critic_states_prime,
            #                                                     org_critic_goals=adver_critic_goals,
            #                                                     u_action=adver_u_actions, c_action=adver_c_actions,
            #                                                     org_rewards=adver_rewards, is_done=adver_terminates)
            #
            #         # train model
            #         if not (abs(adver_elo - agent_elo) > ELO_DIFFRENCE and adver_elo > agent_elo):
            #
            #             # train adversarial models and obtain metrics for each adversarial drone for logging
            #             adver_actor_loss_list, adver_critic_loss_list, adver_actor_grad_norm_list, adver_critic_grad_norm_list = \
            #                 adver_maddpgv2_gnn_agents.apply_gradients_maddpgv2_gnn(num_of_agents=NUMBER_OF_ADVER_DRONES)
            #
            #         else:
            #
            #             adver_actor_loss_list, adver_critic_loss_list, adver_actor_grad_norm_list, adver_critic_grad_norm_list = np.nan, np.nan, np.nan, np.nan
            #
            #     else:
            #
            #         adver_actor_loss_list, adver_critic_loss_list, adver_actor_grad_norm_list, adver_critic_grad_norm_list = np.nan, np.nan, np.nan, np.nan
            #
            # # check if agent is training
            # if AGENT_MODE != "test":
            #
            #     # populate addtional replay buffer for agent maddpgv2_gnn:
            #     if AGENT_MODEL == "maddpgv2_gnn" and agent_maddpgv2_gnn_agents.replay_buffer.org_replay_buffer.is_ep_terminal == True and AGENT_MADDPGV2_GNN_ADDITIONAL_GOALS != 0:
            #         # populate her replay buffer
            #         agent_maddpgv2_gnn_agents.replay_buffer.generate_her_replay_buffer(
            #             opp_org_replay_buffer=adver_maddpgv2_gnn_agents.replay_buffer.org_replay_buffer,
            #             agent_goal=agent_goal,
            #             adver_goal=adver_goal, agent_goal_dist=AGENT_MADDPGV2_GNN_GOAL_DISTRIBUTION,
            #             adver_goal_dist=ADVER_MADDPGV2_GNN_GOAL_DISTRIBUTION)

            # check if adver is training
            if ADVER_MODE != "test":
                pass
                # populate addtional replay buffer for adver maddpgv2_gnn:
                # if ADVER_MODEL == "maddpgv2_gnn" and adver_maddpgv2_gnn_agents.replay_buffer.org_replay_buffer.is_ep_terminal == True and ADVER_MADDPGV2_GNN_ADDITIONAL_GOALS != 0:
                #     # populate her replay buffer
                #     adver_maddpgv2_gnn_agents.replay_buffer.generate_her_replay_buffer(
                #         opp_org_replay_buffer=agent_maddpgv2_gnn_agents.replay_buffer.org_replay_buffer,
                #         agent_goal=agent_goal,
                #         adver_goal=adver_goal, agent_goal_dist=AGENT_MADDPGV2_GNN_GOAL_DISTRIBUTION,
                #         adver_goal_dist=ADVER_MADDPGV2_GNN_GOAL_DISTRIBUTION)

            # metrics logging for agent

            # iterate over num of agent drones
            for i in range(NUMBER_OF_AGENT_DRONES):

                # check if list is not nan and agent model is training
                if np.any(np.isnan(agent_actor_loss_list)) == False and AGENT_MODE != "test":
                    # update sums
                    sum_agent_actor_loss += agent_actor_loss_list[i]

                # check if list is not nan and agent model is training
                if np.any(np.isnan(agent_critic_loss_list)) == False and AGENT_MODE != "test":
                    # update sums
                    sum_agent_critic_loss += agent_critic_loss_list[i]

                # check if list is not nan and agent model is training
                if np.any(np.isnan(agent_actor_grad_norm_list)) == False and AGENT_MODE != "test":
                    # update sums
                    sum_agent_actor_grad_norm += agent_actor_grad_norm_list[i]

                # check if list is not nan and agent model is training
                if np.any(np.isnan(agent_critic_grad_norm_list)) == False and AGENT_MODE != "test":
                    # update sums
                    sum_agent_critic_grad_norm += agent_critic_grad_norm_list[i]

                # check if mappo_gnn
                if AGENT_MODEL == "mappo_gnn":

                    # check if list is not nan and agent model is training
                    if np.any(np.isnan(agent_policy_ratio_list)) == False and AGENT_MODE != "test":
                        # update sums
                        sum_agent_policy_ratio += agent_policy_ratio_list[i]

                # update sum of team and opponent collisions
                sum_agent_number_of_team_collisions += agent_benchmark_data[i, 0]
                sum_agent_number_of_oppo_collisions += agent_benchmark_data[i, 1]

            # metrics logging for adver

            # iterate over num of adver drones
            for i in range(NUMBER_OF_ADVER_DRONES):

                # check if list is not nan and adversarial model is training
                if np.any(np.isnan(adver_actor_loss_list)) == False and ADVER_MODE != "test":
                    # update sums
                    sum_adver_actor_loss += adver_actor_loss_list[i]

                # check if list is not nan and adversarial model is training
                if np.any(np.isnan(adver_critic_loss_list)) == False and ADVER_MODE != "test":
                    # update sums
                    sum_adver_critic_loss += adver_critic_loss_list[i]

                # check if list is not nan and adversarial model is training
                if np.any(np.isnan(adver_actor_grad_norm_list)) == False and ADVER_MODE != "test":
                    # update sums
                    sum_adver_actor_grad_norm += adver_actor_grad_norm_list[i]

                # check if list is not nan and adversarial model is training
                if np.any(np.isnan(adver_critic_grad_norm_list)) == False and ADVER_MODE != "test":
                    # update sums
                    sum_adver_critic_grad_norm += adver_critic_grad_norm_list[i]

                # check if mappo_gnn
                if ADVER_MODEL == "mappo_gnn":

                    # check if list is not nan and adversarial model is training
                    if np.any(np.isnan(
                            adver_policy_ratio_list)) == False and ADVER_MODE != "test" and ADVER_MODEL == "mappo_gnn":
                        # update sums
                        sum_adver_policy_ratio += adver_policy_ratio_list[i]

                # update sum of team and opponent collisions
                sum_adver_number_of_team_collisions += adver_benchmark_data[i, 0]
                sum_adver_number_of_oppo_collisions += adver_benchmark_data[i, 1]

            # log wins for adversarial drones
            if terminal_condition == 1:

                # add sum of wins for adversarial drones
                sum_adver_wins += 1

                # append sums to all lists
                sum_agent_wins_list.append(sum_agent_wins)
                sum_adver_wins_list.append(sum_adver_wins)

                # add opp win and sum of agent win
                writer.add_scalar(tag="terminate_info/agent_drone_win", scalar_value=0, global_step=eps)
                writer.add_scalar(tag="terminate_info/sum_agent_drone_wins", scalar_value=sum_agent_wins,
                                  global_step=eps)

                # add opp win and sum of opp win
                writer.add_scalar(tag="terminate_info/adver_drone_win", scalar_value=1, global_step=eps)
                writer.add_scalar(tag="terminate_info/sum_adver_drone_wins", scalar_value=sum_adver_wins,
                                  global_step=eps)

                # append sums to exceed screen lists
                sum_agent_exceed_screen_list.append(sum_agent_exceed_screen)
                sum_adver_exceed_screen_list.append(sum_adver_exceed_screen)

                # check if experiment terminates if screen exits
                if EXIT_SCREEN_TERMINATE == True:
                    # add agent_exceed_screen and sum_agent_exceed_screen
                    writer.add_scalar(tag="terminate_info/agent_exceed_screen", scalar_value=0, global_step=eps)
                    writer.add_scalar(tag="terminate_info/sum_agent_exceed_screen",
                                      scalar_value=sum_agent_exceed_screen, global_step=eps)

                    # add agent_exceed_screen and sum_agent_exceed_screen
                    writer.add_scalar(tag="terminate_info/adver_exceed_screen", scalar_value=0, global_step=eps)
                    writer.add_scalar(tag="terminate_info/sum_adver_exceed_screen",
                                      scalar_value=sum_adver_exceed_screen, global_step=eps)

                # update agent_goals_softmax_weights for maddpgv2_gnn
                if AGENT_MODEL == "maddpgv2_gnn":
                    update_agent_goals_softmax_weights(agent_goals_softmax_weights=agent_goals_softmax_weights,
                                                       agent_goal_distribution=AGENT_MADDPGV2_GNN_GOAL_DISTRIBUTION,
                                                       agent_elo=agent_elo, adver_elo=adver_elo, agent_goal=agent_goal,
                                                       terminal_condition=terminal_condition)

                # update adver_goals_softmax_weights for maddpgv2_gnn
                if ADVER_MODEL == "maddpgv2_gnn":
                    update_adver_goals_softmax_weights(adver_goals_softmax_weights=adver_goals_softmax_weights,
                                                       adver_goal_distribution=ADVER_MADDPGV2_GNN_GOAL_DISTRIBUTION,
                                                       agent_elo=agent_elo, adver_elo=adver_elo, adver_goal=adver_goal,
                                                       terminal_condition=terminal_condition)

                # calculate elo for agent and adver
                agent_elo, adver_elo = calculate_elo_rating(agent_curr_elo=agent_elo, adver_curr_elo=adver_elo,
                                                            k_agent=AGENT_ELO_K, k_adver=ADVER_ELO_K, d=ELO_D,
                                                            results_list=[terminal_condition],
                                                            results_reward_dict=RESULTS_REWARD_DICT)

                # add agent and adver elo
                writer.add_scalar(tag="elo/agent_elo", scalar_value=agent_elo, global_step=eps)
                writer.add_scalar(tag="elo/adver_elo", scalar_value=adver_elo, global_step=eps)

                # append elo to list
                agent_elo_list.append(agent_elo)
                adver_elo_list.append(adver_elo)

                # iterate over adversarial agents to find disabled
                for agent in env.agents:

                    # check if adver and disabled
                    if agent.adversary == True and agent.movable == False:
                        # increment sum_adver_disabled
                        sum_adver_disabled += 1

                # append sum_adver_disabled to sum_adver_disabled_list
                sum_adver_disabled_list.append(sum_adver_disabled)

                # check scenario
                if SCENARIO_NAME == "zone_def_tag":
                    # add sum_adver_disabled
                    writer.add_scalar(tag="disabled/sum_adver_disabled", scalar_value=sum_adver_disabled,
                                      global_step=eps)

                # print log
                print(f'terminal_condition {terminal_condition}: adversarial drones win')

                # additional log message of goals for maddpgv2_gnn for agent
                if AGENT_MODEL == "maddpgv2_gnn":
                    # print agent_goal
                    print(f'agent_goal (episode time step limit): {agent_goal}')

                # additional log message of goals for maddpgv2_gnn for adversarial
                if ADVER_MODEL == "maddpgv2_gnn":
                    # print adver_goal
                    print(f'adver_goal (restricted radius): {adver_goal}')

            # log wins for agent drones
            elif terminal_condition == 2:

                # add sum of wins for agent drones
                sum_agent_wins += 1

                # append sums to all lists
                sum_agent_wins_list.append(sum_agent_wins)
                sum_adver_wins_list.append(sum_adver_wins)

                # add opp win and sum of agent win
                writer.add_scalar(tag="terminate_info/agent_drone_win", scalar_value=1, global_step=eps)
                writer.add_scalar(tag="terminate_info/sum_agent_drone_wins", scalar_value=sum_agent_wins,
                                  global_step=eps)

                # add opp win and sum of opp win
                writer.add_scalar(tag="terminate_info/adver_drone_win", scalar_value=0, global_step=eps)
                writer.add_scalar(tag="terminate_info/sum_adver_drone_wins", scalar_value=sum_adver_wins,
                                  global_step=eps)

                # append sums to exceed screen lists
                sum_agent_exceed_screen_list.append(sum_agent_exceed_screen)
                sum_adver_exceed_screen_list.append(sum_adver_exceed_screen)

                # check if experiment terminates if screen exits
                if EXIT_SCREEN_TERMINATE == True:
                    # add agent_exceed_screen and sum_agent_exceed_screen
                    writer.add_scalar(tag="terminate_info/agent_exceed_screen", scalar_value=0, global_step=eps)
                    writer.add_scalar(tag="terminate_info/sum_agent_exceed_screen",
                                      scalar_value=sum_agent_exceed_screen, global_step=eps)

                    # add agent_exceed_screen and sum_agent_exceed_screen
                    writer.add_scalar(tag="terminate_info/adver_exceed_screen", scalar_value=0, global_step=eps)
                    writer.add_scalar(tag="terminate_info/sum_adver_exceed_screen",
                                      scalar_value=sum_adver_exceed_screen, global_step=eps)

                # update agent_goals_softmax_weights for maddpgv2_gnn
                if AGENT_MODEL == "maddpgv2_gnn":
                    update_agent_goals_softmax_weights(agent_goals_softmax_weights=agent_goals_softmax_weights,
                                                       agent_goal_distribution=AGENT_MADDPGV2_GNN_GOAL_DISTRIBUTION,
                                                       agent_elo=agent_elo, adver_elo=adver_elo, agent_goal=agent_goal,
                                                       terminal_condition=terminal_condition)

                # update adver_goals_softmax_weights for maddpgv2_gnn
                if ADVER_MODEL == "maddpgv2_gnn":
                    update_adver_goals_softmax_weights(adver_goals_softmax_weights=adver_goals_softmax_weights,
                                                       adver_goal_distribution=ADVER_MADDPGV2_GNN_GOAL_DISTRIBUTION,
                                                       agent_elo=agent_elo, adver_elo=adver_elo, adver_goal=adver_goal,
                                                       terminal_condition=terminal_condition)

                # calculate elo for agent and adver
                agent_elo, adver_elo = calculate_elo_rating(agent_curr_elo=agent_elo, adver_curr_elo=adver_elo,
                                                            k_agent=AGENT_ELO_K, k_adver=ADVER_ELO_K, d=ELO_D,
                                                            results_list=[terminal_condition],
                                                            results_reward_dict=RESULTS_REWARD_DICT)

                # add agent and adver elo
                writer.add_scalar(tag="elo/agent_elo", scalar_value=agent_elo, global_step=eps)
                writer.add_scalar(tag="elo/adver_elo", scalar_value=adver_elo, global_step=eps)

                # append elo to list
                agent_elo_list.append(agent_elo)
                adver_elo_list.append(adver_elo)

                # iterate over adversarial agents to find disabled
                for agent in env.agents:

                    # check if adver and disabled
                    if agent.adversary == True and agent.movable == False:
                        # increment sum_adver_disabled
                        sum_adver_disabled += 1

                # append sum_adver_disabled to sum_adver_disabled_list
                sum_adver_disabled_list.append(sum_adver_disabled)

                # check scenario
                if SCENARIO_NAME == "zone_def_tag":
                    # add sum_adver_disabled
                    writer.add_scalar(tag="disabled/sum_adver_disabled", scalar_value=sum_adver_disabled,
                                      global_step=eps)

                # print log
                print(f'terminal_condition {terminal_condition}: agent drones win')

                # additional log message of goals for maddpgv2_gnn for agent
                if AGENT_MODEL == "maddpgv2_gnn":
                    # print agent_goal
                    print(f'agent_goal (episode time step limit): {agent_goal}')

                # additional log message of goals for maddpgv2_gnn for adversarial
                if ADVER_MODEL == "maddpgv2_gnn":
                    # print adver_goal
                    print(f'adver_goal (restricted radius): {adver_goal}')

            # log agent drones exceeding screen boundary
            elif terminal_condition == 3:

                # add sum_agent_exceed_screen
                sum_agent_exceed_screen += 1

                # append sums to all lists
                sum_agent_wins_list.append(sum_agent_wins)
                sum_adver_wins_list.append(sum_adver_wins)
                sum_agent_exceed_screen_list.append(sum_agent_exceed_screen)
                sum_adver_exceed_screen_list.append(sum_adver_exceed_screen)

                # add opp win and sum of agent win
                writer.add_scalar(tag="terminate_info/agent_drone_win", scalar_value=0, global_step=eps)
                writer.add_scalar(tag="terminate_info/sum_agent_drone_wins", scalar_value=sum_agent_wins,
                                  global_step=eps)

                # add opp win and sum of opp win
                writer.add_scalar(tag="terminate_info/adver_drone_win", scalar_value=0, global_step=eps)
                writer.add_scalar(tag="terminate_info/sum_adver_drone_wins", scalar_value=sum_adver_wins,
                                  global_step=eps)

                # add agent_exceed_screen and sum_agent_exceed_screen
                writer.add_scalar(tag="terminate_info/agent_exceed_screen", scalar_value=1, global_step=eps)
                writer.add_scalar(tag="terminate_info/sum_agent_exceed_screen", scalar_value=sum_agent_exceed_screen,
                                  global_step=eps)

                # add agent_exceed_screen and sum_agent_exceed_screen
                writer.add_scalar(tag="terminate_info/adver_exceed_screen", scalar_value=0, global_step=eps)
                writer.add_scalar(tag="terminate_info/sum_adver_exceed_screen", scalar_value=sum_adver_exceed_screen,
                                  global_step=eps)

                # update agent_goals_softmax_weights for maddpgv2_gnn
                if AGENT_MODEL == "maddpgv2_gnn":
                    update_agent_goals_softmax_weights(agent_goals_softmax_weights=agent_goals_softmax_weights,
                                                       agent_goal_distribution=AGENT_MADDPGV2_GNN_GOAL_DISTRIBUTION,
                                                       agent_elo=agent_elo, adver_elo=adver_elo, agent_goal=agent_goal,
                                                       terminal_condition=terminal_condition)

                # update adver_goals_softmax_weights for maddpgv2_gnn
                if ADVER_MODEL == "maddpgv2_gnn":
                    update_adver_goals_softmax_weights(adver_goals_softmax_weights=adver_goals_softmax_weights,
                                                       adver_goal_distribution=ADVER_MADDPGV2_GNN_GOAL_DISTRIBUTION,
                                                       agent_elo=agent_elo, adver_elo=adver_elo, adver_goal=adver_goal,
                                                       terminal_condition=terminal_condition)

                # calculate elo for agent and adver
                agent_elo, adver_elo = calculate_elo_rating(agent_curr_elo=agent_elo, adver_curr_elo=adver_elo,
                                                            k_agent=AGENT_ELO_K, k_adver=ADVER_ELO_K, d=ELO_D,
                                                            results_list=[terminal_condition],
                                                            results_reward_dict=RESULTS_REWARD_DICT)

                # add agent and adver elo
                writer.add_scalar(tag="elo/agent_elo", scalar_value=agent_elo, global_step=eps)
                writer.add_scalar(tag="elo/adver_elo", scalar_value=adver_elo, global_step=eps)

                # append elo to list
                agent_elo_list.append(agent_elo)
                adver_elo_list.append(adver_elo)

                # iterate over adversarial agents to find disabled
                for agent in env.agents:

                    # check if adver and disabled
                    if agent.adversary == True and agent.movable == False:
                        # increment sum_adver_disabled
                        sum_adver_disabled += 1

                # append sum_adver_disabled to sum_adver_disabled_list
                sum_adver_disabled_list.append(sum_adver_disabled)

                # check scenario
                if SCENARIO_NAME == "zone_def_tag":
                    # add sum_adver_disabled
                    writer.add_scalar(tag="disabled/sum_adver_disabled", scalar_value=sum_adver_disabled,
                                      global_step=eps)

                # print log
                print(f'terminal_condition {terminal_condition}: agent drones exceeded screen boundary')

                # additional log message of goals for maddpgv2_gnn for agent
                if AGENT_MODEL == "maddpgv2_gnn":
                    # print agent_goal
                    print(f'agent_goal (episode time step limit): {agent_goal}')

                # additional log message of goals for maddpgv2_gnn for adversarial
                if ADVER_MODEL == "maddpgv2_gnn":
                    # print adver_goal
                    print(f'adver_goal (restricted radius): {adver_goal}')

            # log adversarial drones exceeding screen boundary
            elif terminal_condition == 4:

                # add sum_adver_exceed_screen
                sum_adver_exceed_screen += 1

                # append sums to all lists
                sum_agent_wins_list.append(sum_agent_wins)
                sum_adver_wins_list.append(sum_adver_wins)
                sum_agent_exceed_screen_list.append(sum_agent_exceed_screen)
                sum_adver_exceed_screen_list.append(sum_adver_exceed_screen)

                # add opp win and sum of agent win
                writer.add_scalar(tag="terminate_info/agent_drone_win", scalar_value=0, global_step=eps)
                writer.add_scalar(tag="terminate_info/sum_agent_drone_wins", scalar_value=sum_agent_wins,
                                  global_step=eps)

                # add opp win and sum of opp win
                writer.add_scalar(tag="terminate_info/adver_drone_win", scalar_value=0, global_step=eps)
                writer.add_scalar(tag="terminate_info/sum_adver_drone_wins", scalar_value=sum_adver_wins,
                                  global_step=eps)

                # add agent_exceed_screen and sum_agent_exceed_screen
                writer.add_scalar(tag="terminate_info/agent_exceed_screen", scalar_value=0, global_step=eps)
                writer.add_scalar(tag="terminate_info/sum_agent_exceed_screen", scalar_value=sum_agent_exceed_screen,
                                  global_step=eps)

                # add agent_exceed_screen and sum_agent_exceed_screen
                writer.add_scalar(tag="terminate_info/adver_exceed_screen", scalar_value=1, global_step=eps)
                writer.add_scalar(tag="terminate_info/sum_adver_exceed_screen", scalar_value=sum_adver_exceed_screen,
                                  global_step=eps)

                # update agent_goals_softmax_weights for maddpgv2_gnn
                if AGENT_MODEL == "maddpgv2_gnn":
                    update_agent_goals_softmax_weights(agent_goals_softmax_weights=agent_goals_softmax_weights,
                                                       agent_goal_distribution=AGENT_MADDPGV2_GNN_GOAL_DISTRIBUTION,
                                                       agent_elo=agent_elo, adver_elo=adver_elo, agent_goal=agent_goal,
                                                       terminal_condition=terminal_condition)

                # update adver_goals_softmax_weights for maddpgv2_gnn
                if ADVER_MODEL == "maddpgv2_gnn":
                    update_adver_goals_softmax_weights(adver_goals_softmax_weights=adver_goals_softmax_weights,
                                                       adver_goal_distribution=ADVER_MADDPGV2_GNN_GOAL_DISTRIBUTION,
                                                       agent_elo=agent_elo, adver_elo=adver_elo, adver_goal=adver_goal,
                                                       terminal_condition=terminal_condition)

                # calculate elo for agent and adver
                agent_elo, adver_elo = calculate_elo_rating(agent_curr_elo=agent_elo, adver_curr_elo=adver_elo,
                                                            k_agent=AGENT_ELO_K, k_adver=ADVER_ELO_K, d=ELO_D,
                                                            results_list=[terminal_condition],
                                                            results_reward_dict=RESULTS_REWARD_DICT)

                # add agent and adver elo
                writer.add_scalar(tag="elo/agent_elo", scalar_value=agent_elo, global_step=eps)
                writer.add_scalar(tag="elo/adver_elo", scalar_value=adver_elo, global_step=eps)

                # append elo to list
                agent_elo_list.append(agent_elo)
                adver_elo_list.append(adver_elo)

                # iterate over adversarial agents to find disabled
                for agent in env.agents:

                    # check if adver and disabled
                    if agent.adversary == True and agent.movable == False:
                        # increment sum_adver_disabled
                        sum_adver_disabled += 1

                # append sum_adver_disabled to sum_adver_disabled_list
                sum_adver_disabled_list.append(sum_adver_disabled)

                # check scenario
                if SCENARIO_NAME == "zone_def_tag":
                    # add sum_adver_disabled
                    writer.add_scalar(tag="disabled/sum_adver_disabled", scalar_value=sum_adver_disabled,
                                      global_step=eps)

                # print log
                print(f'terminal_condition {terminal_condition}: adversarial drones exceeded screen boundary')

                # additional log message of goals for maddpgv2_gnn for agent
                if AGENT_MODEL == "maddpgv2_gnn":
                    # print agent_goal
                    print(f'agent_goal (episode time step limit): {agent_goal}')

                # additional log message of goals for maddpgv2_gnn for adversarial
                if ADVER_MODEL == "maddpgv2_gnn":
                    # print adver_goal
                    print(f'adver_goal (restricted radius): {adver_goal}')

            # update actor_states, adver_actor_states, agent_actor_states
            actor_states = actor_states_prime
            adver_actor_states = adver_actor_states_prime
            agent_actor_states = agent_actor_states_prime

            # reset agent_eps_steps
            if AGENT_MODEL == "mappo_gnn" and agent_eps_steps % AGENT_MAPPO_GNN_EPISODE_LENGTH == 0:
                agent_eps_steps = 0

            # reset adver_eps_steps
            if ADVER_MODEL == "mappo_gnn" and adver_eps_steps % ADVER_MAPPO_GNN_EPISODE_LENGTH == 0:
                adver_eps_steps = 0

        # generate metric logs for agent

        # obtain avg actor and critic loss
        avg_agent_actor_loss = sum_agent_actor_loss / float(NUMBER_OF_AGENT_DRONES * env.world.ep_time_step)
        avg_agent_critic_loss = sum_agent_critic_loss / float(NUMBER_OF_AGENT_DRONES * env.world.ep_time_step)

        # obtain avg actor and critic grad norms
        avg_agent_actor_grad_norm = sum_agent_actor_grad_norm / float(NUMBER_OF_AGENT_DRONES * env.world.ep_time_step)
        avg_agent_critic_grad_norm = sum_agent_critic_grad_norm / float(NUMBER_OF_AGENT_DRONES * env.world.ep_time_step)

        # check if agent model is training
        if AGENT_MODE != "test":
            # add avg actor and critic loss for agent drones
            writer.add_scalar(tag="avg_agent_actor_loss", scalar_value=avg_agent_actor_loss, global_step=eps)
            writer.add_scalar(tag="avg_agent_critic_loss", scalar_value=avg_agent_critic_loss, global_step=eps)

            # add avg actor and critic grad norms for agent drones
            writer.add_scalar(tag="avg_agent_actor_grad_norm", scalar_value=avg_agent_actor_grad_norm, global_step=eps)
            writer.add_scalar(tag="avg_agent_critic_grad_norm", scalar_value=avg_agent_critic_grad_norm,
                              global_step=eps)

        # append avg_agent_actor_loss and avg_agent_critic_loss to their respective list
        avg_agent_actor_loss_list.append(avg_agent_actor_loss)
        avg_agent_critic_loss_list.append(avg_agent_critic_loss)

        # append avg actor and critic grad norms to their respective list
        avg_agent_actor_grad_norm_list.append(avg_agent_actor_grad_norm)
        avg_agent_critic_grad_norm_list.append(avg_agent_critic_grad_norm)

        # obtain avg avg_agent_policy_ratio
        avg_agent_policy_ratio = sum_agent_policy_ratio / float(NUMBER_OF_AGENT_DRONES * env.world.ep_time_step)

        # add avg_agent_policy_ratio for agent drones
        writer.add_scalar(tag="avg_agent_policy_ratio", scalar_value=avg_agent_policy_ratio, global_step=eps)

        # append avg_agent_policy_ratio to list
        avg_agent_policy_ratio_list.append(avg_agent_policy_ratio)

        # add sum team and oppo collisions for agent drones
        writer.add_scalar(tag="sum_agent_number_of_team_collisions", scalar_value=sum_agent_number_of_team_collisions,
                          global_step=eps)
        writer.add_scalar(tag="sum_agent_number_of_oppo_collisions", scalar_value=sum_agent_number_of_oppo_collisions,
                          global_step=eps)

        # append sum_agent_number_of_team_collisisum and sum_agent_number_of_oppo_collisions to their respective list
        sum_agent_number_of_team_collisions_list.append(sum_agent_number_of_team_collisions)
        sum_agent_number_of_oppo_collisions_list.append(sum_agent_number_of_oppo_collisions)

        # obtain avg team and oppo collisions
        avg_agent_number_of_team_collisions = sum_agent_number_of_team_collisions / float(
            NUMBER_OF_AGENT_DRONES * env.world.ep_time_step)
        avg_agent_number_of_oppo_collisions = sum_agent_number_of_oppo_collisions / float(
            NUMBER_OF_AGENT_DRONES * env.world.ep_time_step)

        # add avg team and oppo collisions for agent drones
        writer.add_scalar(tag="avg_agent_number_of_team_collisions", scalar_value=avg_agent_number_of_team_collisions,
                          global_step=eps)
        writer.add_scalar(tag="avg_agent_number_of_oppo_collisions", scalar_value=avg_agent_number_of_oppo_collisions,
                          global_step=eps)

        # append avg_agent_number_of_team_collisions and avg_agent_number_of_oppo_collisions to their respective list
        avg_agent_number_of_team_collisions_list.append(avg_agent_number_of_team_collisions)
        avg_agent_number_of_oppo_collisions_list.append(avg_agent_number_of_oppo_collisions)

        # add agent goal
        writer.add_scalar(tag="goal/agent_goal", scalar_value=agent_goal, global_step=eps)

        # append agent goal to agent_goals_list
        agent_goals_list.append(agent_goal)

        # generate metric logs for adver

        # obtain avg actor and critic loss
        avg_adver_actor_loss = sum_adver_actor_loss / float(NUMBER_OF_ADVER_DRONES * env.world.ep_time_step)
        avg_adver_critic_loss = sum_adver_critic_loss / float(NUMBER_OF_ADVER_DRONES * env.world.ep_time_step)

        # obtain avg actor and critic grad norms
        avg_adver_actor_grad_norm = sum_adver_actor_grad_norm / float(NUMBER_OF_ADVER_DRONES * env.world.ep_time_step)
        avg_adver_critic_grad_norm = sum_adver_critic_grad_norm / float(NUMBER_OF_ADVER_DRONES * env.world.ep_time_step)

        # check if agent model is training
        if ADVER_MODE != "test":
            # add avg actor and critic loss for adver drones
            writer.add_scalar(tag="avg_adver_actor_loss", scalar_value=avg_adver_actor_loss, global_step=eps)
            writer.add_scalar(tag="avg_adver_critic_loss", scalar_value=avg_adver_critic_loss, global_step=eps)

            # add avg actor and critic grad norms for adver drones
            writer.add_scalar(tag="avg_adver_actor_grad_norm", scalar_value=avg_adver_actor_grad_norm, global_step=eps)
            writer.add_scalar(tag="avg_adver_critic_grad_norm", scalar_value=avg_adver_critic_grad_norm,
                              global_step=eps)

        # append avg_adver_actor_loss and avg_adver_critic_loss to their respective list
        avg_adver_actor_loss_list.append(avg_adver_actor_loss)
        avg_adver_critic_loss_list.append(avg_adver_critic_loss)

        # append avg actor and critic grad norms to their respective list
        avg_adver_actor_grad_norm_list.append(avg_adver_actor_grad_norm)
        avg_adver_critic_grad_norm_list.append(avg_adver_critic_grad_norm)

        # obtain avg avg_adver_policy_ratio
        avg_adver_policy_ratio = sum_adver_policy_ratio / float(NUMBER_OF_ADVER_DRONES * env.world.ep_time_step)

        # add avg_adver_policy_ratio for agent drones
        writer.add_scalar(tag="avg_adver_policy_ratio", scalar_value=avg_adver_policy_ratio, global_step=eps)

        # append avg_adver_policy_ratio to list
        avg_adver_policy_ratio_list.append(avg_adver_policy_ratio)

        # add sum team and oppo collisions for adver drones
        writer.add_scalar(tag="sum_adver_number_of_team_collisions", scalar_value=sum_adver_number_of_team_collisions,
                          global_step=eps)
        writer.add_scalar(tag="sum_adver_number_of_oppo_collisions", scalar_value=sum_adver_number_of_oppo_collisions,
                          global_step=eps)

        # append sum_adver_number_of_team_collisisum and sum_adver_number_of_oppo_collisions to their respective list
        sum_adver_number_of_team_collisions_list.append(sum_adver_number_of_team_collisions)
        sum_adver_number_of_oppo_collisions_list.append(sum_adver_number_of_oppo_collisions)

        # obtain avg team and oppo collisions
        avg_adver_number_of_team_collisions = sum_adver_number_of_team_collisions / float(
            NUMBER_OF_ADVER_DRONES * env.world.ep_time_step)
        avg_adver_number_of_oppo_collisions = sum_adver_number_of_oppo_collisions / float(
            NUMBER_OF_ADVER_DRONES * env.world.ep_time_step)

        # add avg team and oppo collisions for adver drones
        writer.add_scalar(tag="avg_adver_number_of_team_collisions", scalar_value=avg_adver_number_of_team_collisions,
                          global_step=eps)
        writer.add_scalar(tag="avg_adver_number_of_oppo_collisions", scalar_value=avg_adver_number_of_oppo_collisions,
                          global_step=eps)

        # append avg_adver_number_of_team_collisions and avg_adver_number_of_oppo_collisions to their respective list
        avg_adver_number_of_team_collisions_list.append(avg_adver_number_of_team_collisions)
        avg_adver_number_of_oppo_collisions_list.append(avg_adver_number_of_oppo_collisions)

        # add adver goal
        writer.add_scalar(tag="goal/adver_goal", scalar_value=adver_goal, global_step=eps)

        # append adver goal to adver_goals_list
        adver_goals_list.append(adver_goal)

        # check if metrics is to be saved in csv log
        if SAVE_CSV_LOG == True:
            # generate pandas dataframe to store logs
            df = pd.DataFrame(list(
                zip(list(range(1, eps + 1)), sum_agent_wins_list, sum_adver_wins_list, sum_agent_exceed_screen_list,
                    sum_adver_exceed_screen_list,
                    sum_agent_number_of_team_collisions_list, sum_agent_number_of_oppo_collisions_list,
                    sum_adver_number_of_team_collisions_list,
                    sum_adver_number_of_oppo_collisions_list, sum_adver_disabled_list,
                    avg_agent_number_of_team_collisions_list, avg_agent_number_of_oppo_collisions_list,
                    avg_adver_number_of_team_collisions_list, avg_adver_number_of_oppo_collisions_list,
                    avg_agent_actor_loss_list, avg_agent_critic_loss_list,
                    avg_adver_actor_loss_list, avg_adver_critic_loss_list, avg_agent_actor_grad_norm_list,
                    avg_agent_critic_grad_norm_list, avg_adver_actor_grad_norm_list,
                    avg_adver_critic_grad_norm_list, avg_agent_policy_ratio_list, avg_adver_policy_ratio_list,
                    agent_elo_list, adver_elo_list, agent_goals_list, adver_goals_list)),
                columns=['episodes', 'sum_agent_wins', 'sum_adver_wins', 'sum_agent_exceed_screen',
                         'sum_adver_exceed_screen',
                         'sum_agent_number_of_team_collisions', 'sum_agent_number_of_oppo_collisions',
                         'sum_adver_number_of_team_collisions',
                         'sum_adver_number_of_oppo_collisions', 'sum_adver_disabled',
                         'avg_agent_number_of_team_collisions', 'avg_agent_number_of_oppo_collisions',
                         'avg_adver_number_of_team_collisions', 'avg_adver_number_of_oppo_collisions',
                         'avg_agent_actor_loss', 'avg_agent_critic_loss',
                         'avg_adver_actor_loss', 'avg_adver_critic_loss', 'avg_agent_actor_grad_norm',
                         'avg_agent_critic_grad_norm', 'avg_adver_actor_grad_norm',
                         'avg_adver_critic_grad_norm', 'avg_agent_policy_ratio_list',
                         'avg_adver_policy_ratio_list', 'agent_elo', 'adver_elo', 'agent_goal',
                         'adver_goal'])

            # store training logs
            df.to_csv(
                CSV_LOG_DIRECTORY + '/' + GENERAL_TRAINING_NAME + "_" + AGENT_MODE + "_" + ADVER_MODE + "_logs.csv",
                index=False)

        # check if agent is training and at correct episode to save
        if AGENT_MODE != "test" and eps % SAVE_MODEL_RATE == 0:

            # check if agent model is maddpg_gnn
            # if AGENT_MODEL == "maddpg_gnn":
            #
            #     # save all models
            #     agent_maddpg_gnn_agents.save_all_models()

            # check if agent model is mappo_gnn
            if AGENT_MODEL == "mappo_gnn":
                # save all models
                agent_mappo_gnn_agents.save_all_models()

            # check if agent model is maddpgv2_gnn
            # elif AGENT_MODEL == "maddpgv2_gnn":
            #
            #     # save all models
            #     agent_maddpgv2_gnn_agents.save_all_models()

        # check if adver is training and at correct episode to save
        if ADVER_MODE != "test" and eps % SAVE_MODEL_RATE == 0:

            # check if adver model is maddpg_gnn
            # if ADVER_MODEL == "maddpg_gnn":
            #
            #     # save all models
            #     adver_maddpg_gnn_agents.save_all_models()

            # check if adver model is mappo_gnn
            if ADVER_MODEL == "mappo_gnn":
                # save all models
                adver_mappo_gnn_agents.save_all_models()

            # check if adver model is maddpgv2_gnn
            # elif ADVER_MODEL == "maddpgv2_gnn":
            #
            #     # save all models
            #     adver_maddpgv2_gnn_agents.save_all_models()


if __name__ == "__main__":
    train_test()
