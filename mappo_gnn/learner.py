import datetime
import time
import os

import numpy as np
import tensorflow as tf
from smac.env import StarCraft2Env

from parameters import *
from mappo_gnn import mappo_gnn
from runner import Runner
from util import get_session, save_state
from torch.utils.tensorboard import SummaryWriter


def learn(env):
    """Update model and record performance."""
    evalue_env = StarCraft2Env(**env_args)
    evalue_env._seed=env_args['seed'] + 100 * N_ENVS

    sess = get_session()
    
    # Recording alg and env setting
    folder_name = 'summaries/' + SCENARIO_NAME + datetime.datetime.now().strftime('%d-%m-%y%H%M')
    tb=SummaryWriter(folder_name)
    txt_path = folder_name + '/' + SCENARIO_NAME + '.txt'
    with open(txt_path, "w") as file:
        #TODO
        file.write(str(env_args))
        file.write(str(alg_args))

    model = mappo_gnn(env=env)
    runner = Runner(env=env, model=model)

    num_episodes = 0
    last_evaluation_t = -EVALUE_INTERVAL - 1
    start_time = time.perf_counter()

    for update in range(1, N_UPDATES + 1):
        # Get experience from multiple environments
        obs, state, returns, values, actions, ps, mb_critic_hidden_state_c, mb_critic_hidden_state_h, \
         mb_actor_hidden_state_c, mb_actor_hidden_state_h, ep_infos, performance_r = runner.run()

        # Training
        mb_loss = []
        inds = np.arange(BATCH_SIZE)
        for _ in range(N_EPOCHS):
            np.random.shuffle(inds)
            # 0 to batch_size with batch_train_size step
            for start in range(0, BATCH_SIZE, MINIBATCH_SIZE):
                end = start + MINIBATCH_SIZE
                mb_inds = inds[start:end]
                slices = (arr[mb_inds] for arr in
                          (obs, state, returns, values, actions, ps, mb_critic_hidden_state_c,
                           mb_critic_hidden_state_h, mb_actor_hidden_state_c, mb_actor_hidden_state_h))
                mb_loss.append(model.train(*slices))

        # Recording
        loss_vals = np.nanmean(mb_loss, axis=0)
        num_episodes += len(ep_infos)
        # summary_episodes = tf.Summary() 
        # summary_step = tf.Summary()

        performance_dead_al = []
        performance_dead_enemies = []
        performance_won_rate = []
        for items in ep_infos:
            if len(items) != 3:
                continue
            else:
                performance_dead_al.append(items['dead_allies'])
                performance_dead_enemies.append(items['dead_enemies'])
                performance_won_rate.append(items['battle_won'])

        performance_dead_al = np.nanmean(performance_dead_al)
        performance_dead_enemies = np.nanmean(performance_dead_enemies)
        performance_won_rate = np.nanmean(performance_won_rate)

        tb.add_scalar(tag='Perf/Reward', scalar_value = performance_r)
        tb.add_scalar(tag='Perf/Dead_allies', scalar_value = performance_dead_al)
        tb.add_scalar(tag='Perf/Dead_enemies', scalar_value = performance_dead_enemies)
        tb.add_scalar(tag='Perf/Won_rate', scalar_value = performance_won_rate)
        tb.add_scalar(tag='Perf_Step/Reward_Step', scalar_value = performance_r)
        tb.add_scalar(tag='Perf_Step/Dead_allies', scalar_value = performance_dead_al)
        tb.add_scalar(tag='Perf_Step/Dead_enemies', scalar_value = performance_dead_enemies)
        tb.add_scalar(tag='Perf_Step/Won_rate', scalar_value = performance_won_rate)

        for (val, name) in zip(loss_vals, model.loss_names):
            if name == 'actor_grad_norm' or name == 'critic_grad_norm':
                tb.add_scalar(tag='grad/' + name, scalar_value = val)
                tb.add_scalar(tag='grad_step/' + name, scalar_value = val)
            else:
                tb.add_scalar(tag='loss/' + name, scalar_value = val)
                tb.add_scalar(tag='loss_step/' + name, scalar_value = val)

        tb.add_scalar(tag='num_episodes', scalar_value = num_episodes)
        tb.add_scalar(tag='step_episodes', scalar_value = update * BATCH_SIZE)
        tb.flush()

        if update % 10 == 0:
            print('update: {}, episode: {}, step: {}, episode reward: {},'
                  'dead allies: {}, dead enemies: {}, won rate: {}'.format(update, num_episodes,
                                                                         update * BATCH_SIZE, performance_r,
                                                                         performance_dead_al,
                                                                         performance_dead_enemies,
                                                                         performance_won_rate))

        if (update * BATCH_SIZE - last_evaluation_t) / EVALUE_INTERVAL >= 1.0:
            # Evaluate model
            last_evaluation_t = update * BATCH_SIZE
            eval_reward, eval_dead_allies, eval_dead_enemies, eval_ep_len, eval_win_rate = evaluate(evalue_env, model)
            # Recording
            tb.add_scalar(tag='Perf_evaluate_step/Reward', scalar_value = eval_reward)
            tb.add_scalar(tag='Perf_evaluate_step/Dead_allies', scalar_value = eval_dead_allies)
            tb.add_scalar(tag='Perf_evaluate_step/Dead_enemies', scalar_value = eval_dead_enemies)
            tb.add_scalar(tag='Perf_evaluate_step/Episode_len', scalar_value = eval_ep_len)
            tb.add_scalar(tag='Perf_evaluate_step/Won_rate', scalar_value = eval_win_rate)
            tb.add_scalar(tag='Perf_evaluate_episode/Reward', scalar_value = eval_reward)
            tb.add_scalar(tag='Perf_evaluate_episode/Dead_allies', scalar_value = eval_dead_allies)
            tb.add_scalar(tag='Perf_evaluate_episode/Dead_enemies', scalar_value = eval_dead_enemies)
            tb.add_scalar(tag='Perf_evaluate_episode/Episode_len', scalar_value = eval_ep_len)
            tb.add_scalar(tag='Perf_evaluate_episode/Won_rate', scalar_value = eval_win_rate)
            tb.add_scalar(tag='num_episodes_eval', scalar_value = num_episodes)
            tb.add_scalar(tag='step_episodes_eval', scalar_value = update * BATCH_SIZE)
            tb.flush()

        if update % SAVE_INTERVAL == 0:
            t_now = time.perf_counter()
            print('consume time', t_now - start_time)
            save_path = "my_model/" + SCENARIO_NAME+"/" + '%.5i' % update
            os.makedirs(save_path, exist_ok=True)
            print('Saving to', save_path)
            save_path += "/"+'%.5i' % update
            save_state(save_path)

    evalue_env.close()


def evaluate(evalue_env, model):
    """Evaluate model."""
    eval_reward = []
    eval_dead_allies = []
    eval_dead_enemies = []
    eval_ep_len = []
    eval_win_rate = []
    for _ in range(EVALUE_EPISODES):
        evalue_env.reset()
        terminal = False
        episode_reward = 0
        ep_len = 0
        #todo: hidden dims
        while not terminal:
            obs = evalue_env.get_obs()
            obs = np.expand_dims(obs, axis=0)
            valid_action = evalue_env.get_avail_actions()
            actions, actor_hidden_state = model.evalue(obs, valid_action, actor_hidden_state_c, actor_hidden_state_h)
            actor_hidden_state_c, actor_hidden_state_h = actor_hidden_state
            reward, terminal, info = evalue_env.step(np.squeeze(actions))
            ep_len += 1
            episode_reward += reward
            if terminal:
                eval_reward.append(episode_reward)
                eval_ep_len.append(ep_len)
                if len(info) != 3:
                    continue
                else:
                    eval_dead_allies.append(info['dead_allies'])
                    eval_dead_enemies.append(info['dead_enemies'])
                    eval_win_rate.append(info['battle_won'])
    eval_reward = np.nanmean(eval_reward)
    eval_dead_allies = np.nanmean(eval_dead_allies)
    eval_dead_enemies = np.nanmean(eval_dead_enemies)
    eval_ep_len = np.nanmean(eval_ep_len)
    eval_win_rate = np.nanmean(eval_win_rate)
    return eval_reward, eval_dead_allies, eval_dead_enemies, eval_ep_len, eval_win_rate
