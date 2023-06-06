# Bipedal walker using SAC with hints

This is the directory containing the code for SAC agent training with or without hints. The hints are provided by another agent, which is initialized by a saved model. This saved model is first created by running SAC, without hints, to solve the ```BipedalWalker-v3``` environment.

<img src="./movie.gif" alt="trained BipedalWalkerHardcore" width=400>

The steps to perform are:

  * Train an agent for ```BipedalWalker-v3``` as

  ```
  python main.py --seed 2 --env_name BipedalWalker-v3 --iteration 1500
  ```

  * Copy the saved models to use later to provide hints

  ```
  cp learnerq_eval_1_sac_critic.model hinterq_eval_1_sac_critic.model

  cp learnerq_eval_2_sac_critic.model hinterq_eval_2_sac_critic.model

  cp learnera_eval_sac_actor.model hintera_eval_sac_actor.model

  cp learnerreplaymem_sac.model hinterreplaymem_sac.model
  ```

  * Now you can use hints, for example to solve the ```BipedalWalkerHardcore-v3``` environment

  ```
  python main.py --seed 4 --env_name BipedalWalkerHardcore-v3 --iteration 3500 --use_hint
  ```

You can change the random seeds as you like.

Files provided are:

```main.py``` : main method, use ```NEW_GYM=True``` with new gym 0.26.2

```net_sac.py``` : networks, SAC agent, learning using ADMM

```PER.py``` : prioritized experience replay memory (not used)

```display.py``` : disply the agent in action
