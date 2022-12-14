# ==========================================================================================================================================================
# mappo gnn agent class 
# purpose: creates and updates neural network 
# ==========================================================================================================================================================

import torch as T

from utils.nn import mappo_dgcn_actor_model, mappo_dgcn_critic_model


class mappo_gnn_agent:

    def __init__(self, mode, scenario_name, training_name, lr_actor, lr_critic, optimizer, lr_scheduler, obs_dims,
                 dgcn_output_dims, somu_lstm_hidden_size, somu_lstm_num_layers, somu_lstm_dropout, num_somu_lstm,
                 scmu_lstm_hidden_size, scmu_lstm_num_layers, scmu_lstm_dropout, num_scmu_lstm, somu_multi_att_num_heads,
                 somu_multi_att_dropout, scmu_multi_att_num_heads, scmu_multi_att_dropout, actor_fc_output_dims,
                 actor_fc_dropout_p, softmax_actions_dims, softmax_actions_dropout_p, gatv2_input_dims, 
                 gatv2_output_dims, gatv2_num_heads, gatv2_dropout_p, gatv2_bool_concat, gmt_hidden_dims, gmt_output_dims, 
                 critic_fc_output_dims, critic_fc_dropout_p):

        """ class constructor for mappo agent attributes """

        # learning rate for actor model
        self.lr_actor = lr_actor

        # learning rate for critic model
        self.lr_critic = lr_critic

        # intialise actor model 
        self.mappo_gnn_actor = mappo_dgcn_actor_model(model="mappo_gnn_actor", model_name=None, mode=mode,
                                                      scenario_name=scenario_name, training_name=training_name,
                                                      learning_rate=self.lr_actor, optimizer=optimizer, 
                                                      lr_scheduler=lr_scheduler, obs_dims=obs_dims,
                                                      dgcn_output_dims=dgcn_output_dims,
                                                      somu_lstm_hidden_size=somu_lstm_hidden_size,
                                                      somu_lstm_num_layers=somu_lstm_num_layers,
                                                      somu_lstm_dropout=somu_lstm_dropout, num_somu_lstm=num_somu_lstm,
                                                      scmu_lstm_hidden_size=scmu_lstm_hidden_size,
                                                      scmu_lstm_num_layers=scmu_lstm_num_layers,
                                                      scmu_lstm_dropout=scmu_lstm_dropout,
                                                      num_scmu_lstm=num_scmu_lstm,
                                                      somu_multi_att_num_heads=somu_multi_att_num_heads,
                                                      somu_multi_att_dropout=somu_multi_att_dropout,
                                                      scmu_multi_att_num_heads=scmu_multi_att_num_heads,
                                                      scmu_multi_att_dropout=scmu_multi_att_dropout,
                                                      actor_fc_output_dims=actor_fc_output_dims,
                                                      actor_fc_dropout_p=actor_fc_dropout_p,
                                                      softmax_actions_dims=softmax_actions_dims,
                                                      softmax_actions_dropout_p=softmax_actions_dropout_p
                                                      )
       
        # intialise critic model
        self.mappo_gnn_critic = mappo_dgcn_critic_model(model="mappo_gnn_critic", model_name=None, mode=mode,
                                                        scenario_name=scenario_name, training_name=training_name,
                                                        learning_rate=self.lr_critic, optimizer=optimizer, 
                                                        lr_scheduler=lr_scheduler, gatv2_input_dims = gatv2_input_dims, 
                                                        gatv2_output_dims = gatv2_output_dims, gatv2_num_heads = gatv2_num_heads, 
                                                        gatv2_dropout_p = gatv2_dropout_p, gatv2_bool_concat = gatv2_bool_concat, 
                                                        gmt_hidden_dims = gmt_hidden_dims, gmt_output_dims = gmt_output_dims, 
                                                        critic_fc_output_dims = critic_fc_output_dims, 
                                                        critic_fc_dropout_p = critic_fc_dropout_p)

    def select_action(self, mode, state):

        """ function to select action for the agent given state observed by local agent """

        # set actor model to evaluation mode (for batch norm and dropout) --> remove instances of batch norm, dropout etc. (things that shd only be around in training)
        self.mappo_gnn_actor.eval()

        # turn actor local state observations to tensor in actor device
        actor_input = T.tensor(state, dtype=T.float).to(self.mappo_gnn_actor.device)

        # add batch dimension to inputs
        actor_input = actor_input.unsqueeze(0)

        # feed actor_input to obtain motor and communication actions 
        softmax_actions = self.mappo_gnn_actor.forward(actor_input)

        # set actor model to training mode (for batch norm and dropout)
        self.mappo_gnn_actor.train()

        # sample from distribution if not test
        if mode != "test":

            # obtain distribution
            actions_dist = T.distributions.categorical.Categorical(softmax_actions)

            # obtain sample
            action_sample = actions_dist.sample()

            # obtain log_prob
            log_prob = actions_dist.log_prob(action_sample)

            return action_sample.cpu().detach().numpy()[0], action_log_prob.cpu().detach().numpy()[0]

        # take argmax
        else:

            return T.argmax(softmax_actions, dim = -1).cpu().detach().numpy()[0], None

    def save_models(self):

        """ save weights """

        # save weights for each actor, target_actor, critic, target_critic model
        T.save(self.mappo_gnn_actor.state_dict(), self.mappo_gnn_actor.checkpoint_path)
        T.save(self.mappo_gnn_critic.state_dict(), self.mappo_gnn_critic.checkpoint_path)

    def load_models(self):

        """ load weights """

        # load weights for each actor, target_actor, critic, target_critic model
        self.mappo_gnn_actor.load_state_dict(T.load(self.mappo_gnn_actor.checkpoint_path, map_location=T.device(
            'cuda:0' if T.cuda.is_available() else 'cpu')))
        self.mappo_gnn_critic.load_state_dict(T.load(self.mappo_gnn_critic.checkpoint_path, map_location=T.device(
            'cuda:0' if T.cuda.is_available() else 'cpu')))
