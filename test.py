# -*- coding: utf-8 -*-
import argparse
import mario_env
from stable_baselines3 import PPO

def play(model_path):
    env = mario_env.create_env()
    model = PPO.load(model_path)
    state = env.reset()

    while True:
        action, _states = model.predict(state)
        state, reward, done, info = env.step(action)
        env.render()
        if done:
            state = env.reset()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a trained PPO model on Super Mario Bros")
    parser.add_argument('--model_path', default="./best_model", type=str, required=False, help='Path to the trained model')
    args = parser.parse_args()

    play(args.model_path)