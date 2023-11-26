from __future__ import absolute_import, division, print_function
from copy import deepcopy

import os
import logging
import tensorflow as tf
import sys

from tf_agents.environments import gym_wrapper
from tf_agents.environments import TFPyEnvironment
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents import agents
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.policies import greedy_policy, policy_saver, random_tf_policy
from tf_agents.drivers import dynamic_step_driver
from tf_agents.utils import common


from src.rl.environment.environment import EnvironmentBuilder


class ObservationPreprocessingLayer(tf.keras.layers.Layer):
    """
    Layer that proccess enviroment observation format (dict)
    and tranform to a unidimensional tensor
    """

    def __init__(self):
        super(ObservationPreprocessingLayer, self).__init__()

    def call(self, inputs):
        # Combine or preprocess inputs here
        combined_inputs = tf.concat(list(inputs.values()), axis=-1)
        return combined_inputs


class TF_DQN_PIPELINE:
    DEFAULT_METRICS = {
        "eval_freq": 100,
        "gamma": 0.9,
        "learning_rate": 0.01,
        "learning_starts": 150,
        "eie": 1,
        "efe": 0.6,
        "ef": 0.1,
        "tau": 0.1,
        "buffer_size": 50000,
        "batch_size": 500,
        "collect_steps_per_iteration": 1,
        "initial_collect_steps": 10,
        "target_update_tau": 1,
        "target_update_interval": 50,
        "n_steps": 10000,
        "log_interval": 100,
        "checkpoint_callback": 100,
        "output_fc_layer_params": (100,),
        "eie": 1,
        "percentage_of_it_decay": 0.65,
        "efe": 0.05,
    }

    def load_config(self):
        # Override the load config method to extend it's funtionlity
        # and support default agent training configs
        self.params = deepcopy(TF_DQN_PIPELINE.DEFAULT_METRICS)
        # update the default params with the loaded configuration
        return self.params

    def build_enviroment_instance(self):
        gym_env = env = EnvironmentBuilder(
        ticket_path='data/data/test_ticket.csv',
        planogram_csv_path='data/data/planogram_table.csv',
        customer_properties_path='data/data/hackathon_customers_properties.csv',
        grouping_path='data/data/article_group.json',
        obs_mode=None,
        reward_mode=1).build()
        self.constraint_splitter = env.observation_and_action_constrain_splitter
        # wrapps the gym enviroment to PythonEnviroment
        return TFPyEnvironment(
            gym_wrapper.GymWrapper(gym_env)
        )

    def build_callbacks(self):
        # configure agents info metrics logs
        self.__configure_training_info_logs__()
        # build train evaluation metrics callbacks
        self.train_metrics = [
            tf_metrics.NumberOfEpisodes(),
            tf_metrics.EnvironmentSteps(),
            tf_metrics.AverageEpisodeLengthMetric(),
            tf_metrics.AverageReturnMetric()
        ]
        return self.train_metrics

    def init_agent(self):
        self.train_env = self.build_enviroment_instance()
        # create exploration/explotation policy
        self.global_step = tf.Variable(0, dtype=tf.int64, name="global_step", trainable=False)
        # compute the number of decay steps
        if self.params['percentage_of_it_decay']:
            self.n_decacy_steps = self.params['n_steps'] * self.params['percentage_of_it_decay']
        else:
            self.n_decacy_steps = self.params["ef"]
        self.epsilon = tf.compat.v1.train.polynomial_decay(
            self.params["eie"],
            self.global_step,
            self.n_decacy_steps,
            end_learning_rate=self.params["efe"]
        )
        # create q_network
        preprocessing_combiner = ObservationPreprocessingLayer()
        q_net = q_network.QNetwork(
            input_tensor_spec=self.train_env.time_step_spec().observation,
            action_spec=self.train_env.action_spec(),
            preprocessing_layers=None,
            #preprocessing_combiner=preprocessing_combiner,
            conv_layer_params=None,
            fc_layer_params=self.params["output_fc_layer_params"],
            dropout_layer_params=None,
            activation_fn=tf.keras.activations.relu,
            kernel_initializer=None,
            batch_squash=True,
            dtype=tf.float32,
            name="QNetwork",
        )
        # create agent
        self.agent = agents.DqnAgent(
            time_step_spec=self.train_env.time_step_spec(),
            action_spec=self.train_env.action_spec(),
            q_network=q_net,
            epsilon_greedy=self.epsilon,
            observation_and_action_constraint_splitter=self.constraint_splitter,
            target_update_tau=self.params['target_update_tau'],
            target_update_period=self.params['target_update_interval'],
            optimizer=tf.optimizers.Adam(learning_rate=self.params['learning_rate']),
            td_errors_loss_fn=common.element_wise_squared_loss,
            gamma=self.params['gamma'],
            train_step_counter=self.global_step
        )
        self.agent.initialize()

    def learn(self):
        # replay buffer initialization
        (
            replay_buffer,
            collect_driver,
            initial_collect_driver,
            dataset_iterator
        ) = self.__build_init_replay_buffer__()
        # build the checkpointer
        self.__init_checkpointer__()
        # start training proccess
        self.agent.train = common.function(self.agent.train)
        time_step = None
        for t in range(self.params['n_steps']):
            # collect data
            time_step, _ = collect_driver.run(time_step=time_step)
            # run train step
            experience, _ = next(dataset_iterator)
            loss = self.agent.train(experience=experience).loss
            # logging training evolution
            step = self.agent.train_step_counter.numpy()
            tf.summary.scalar('epsilon', self.epsilon(), step=self.agent.train_step_counter)

            if step % self.params['log_interval'] == 0:
                logging.info(f'step {step} loss={loss}')
                for train_metric in self.train_metrics:
                    train_metric.tf_summaries(train_step=self.global_step, step_metrics=self.train_metrics[:2])
            if step % self.params['checkpoint_callback'] == 0:
                self.checkpointer.save(self.global_step)

    def __build_init_replay_buffer__(self):
        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self.agent.collect_data_spec,
            batch_size=self.train_env.batch_size,
            max_length=self.params['buffer_size'],
        )

        self.replay_buffer = replay_buffer

        collect_driver = dynamic_step_driver.DynamicStepDriver(
            env=self.train_env,
            policy=self.agent.collect_policy,
            observers=[replay_buffer.add_batch] + self.train_metrics,
            num_steps=self.params['collect_steps_per_iteration']
        )

        initial_collect_policy = random_tf_policy.RandomTFPolicy(
            time_step_spec=self.train_env.time_step_spec(),
            action_spec=self.train_env.action_spec(),
            observation_and_action_constraint_splitter=self.constraint_splitter
        )

        initial_collect_driver = dynamic_step_driver.DynamicStepDriver(
            env=self.train_env,
            policy=initial_collect_policy, observers=[replay_buffer.add_batch],
            num_steps=self.params['initial_collect_steps']
        )
        # collect initial replay data
        if self.global_step == 0 or self.replay_buffer.num_frames() == 0:
            initial_collect_driver.run()
        # create a data set structure from replay buffer
        dataset = replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=self.params['batch_size'],
            num_steps=2
        )
        return replay_buffer, collect_driver, initial_collect_driver, iter(dataset)

    def __configure_training_info_logs__(self):
        train_summary_writer = tf.summary.create_file_writer(
         'results', flush_millis=10000)
        train_summary_writer.set_as_default()

    def __init_checkpointer__(self):
        checkpoint_dir = 'checkpoints'
        self.checkpointer = common.Checkpointer(
            ckpt_dir=checkpoint_dir,
            max_to_keep=1,
            agent=self.agent,
            policy=self.agent.policy,
            replay_buffer=self.replay_buffer,
            global_step=self.global_step
        )


# main run code
if __name__ == '__main__':
    dqn = TF_DQN_PIPELINE()
    dqn.load_config()
    dqn.init_agent()
    dqn.build_callbacks()
    dqn.learn()