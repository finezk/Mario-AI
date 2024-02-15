# -*- coding: utf-8 -*-
import argparse
import os
import mario_env
import gc
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)
            gc.collect()
        return True

def main(TOTAL_TIMESTEPS, LEARNING_RATE):
    env = mario_env.create_env()

    CHECKPOINT_DIR = './train/'
    LOG_DIR = './logs/'

    callback = TrainAndLoggingCallback(check_freq=50000, save_path=CHECKPOINT_DIR)
    model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=LEARNING_RATE,
                n_steps=256)

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a PPO agent to play Super Mario Bros")
    parser.add_argument('--total_timesteps', type=int, default=1000000, help='Total timesteps for training')
    parser.add_argument('--learning_rate', type=float, default=0.000001, help='Learning rate for the model')
    args = parser.parse_args()

    main(args.total_timesteps, args.learning_rate)