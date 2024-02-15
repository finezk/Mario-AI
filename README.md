
# Super Mario Bros AI

## Introduction
![banner.png (1918×1080) (raw.githubusercontent.com)](https://raw.githubusercontent.com/finezk/alien-invasion/main/banner.png)

Welcome to MarioRL, an exciting project that combines the nostalgia of the classic Super Mario Bros game with the cutting-edge field of reinforcement learning (RL). This project is designed to provide a hands-on experience with RL concepts by training an agent to play Super Mario Bros autonomously. Utilizing the PyTorch library and the Gym Retro environment, MarioRL offers a unique platform for both beginners and experienced practitioners to explore and experiment with various reinforcement learning algorithms in a fun and familiar setting.

## What is MarioRL?

MarioRL is a reinforcement learning framework that uses the iconic Super Mario Bros game as a testing ground for RL agents. The main goal of this project is to demonstrate how an AI agent can learn complex tasks, such as navigating through levels, avoiding obstacles, and defeating enemies, through the process of trial and error. By interacting with the game environment, the agent learns to make decisions that maximize its overall performance, measured by the game score and progress.

## Features

- **Customizable Training Environment:** The project is built on the Gym Retro interface, allowing users to customize game scenarios, levels, and difficulty to tailor the learning experience.
- **State-of-the-Art RL Techniques:** Incorporates various reinforcement learning algorithms, including but not limited to Q-learning, Deep Q-Networks (DQN), and Policy Gradient methods, providing a comprehensive toolkit for RL experimentation.
- **Detailed Performance Metrics:** Comes equipped with tools for monitoring the agent's performance, including score tracking, progress visualization, and learning rate analysis, to help users understand and improve their models.
- **Community and Support:** As an open-source project, MarioRL encourages contributions and discussions. Whether you're looking to improve the framework, share your own Mario-playing agent, or need help troubleshooting, the MarioRL community is here to support you.

## Getting Started

Embark on your reinforcement learning journey with MarioRL by cloning the repository, setting up your environment, and running the provided Jupyter notebook. Whether your interest lies in the technical aspects of RL algorithms or you're just here for the thrill of watching Mario conquer levels on his own, MarioRL promises a rewarding experience.
```
git clone https://github.com/finezk/Mario-AI.git
```


## Usage

1. Visual Studio Code
Download VSCode https://code.visualstudio.com/
Download python 3.10.11 https://www.python.org/

After installing VScode and Python, download the extension in VSCode as follow:
* python
* python for vscode
* code runner

2. Open the terminal in the "Mario-AI" folder.
```
pip install -r requirements.txt
```
3. How to train model
```
cd Mario-AI
python train.py
```
4. To test the AI model, drag the file you want to the "Mario-AI" layer of the folder and rename it "best_model."   
```
cd Mario-AI
python test.py
```      

##  Reference
https://www.youtube.com/watch?v=2eeYqJ0uBKE