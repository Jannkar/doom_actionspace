# Stable Baselines agent
Codes to train the Stable Baselines agent. The source model can be trained with DQN and PPO using replace and adapter methods.

## Training
### Source model
Train the source Doom agent (DQN or PPO):  
`python3 doom_baselines_agents.py --mode train --algorithm [DQN/PPO] --environment doom --time_steps 1000000`

Most important arguments for training the source model:
* `--env` - Use "doom", "sim-to-sim" or "turtlebot" accordingly
* `--algorithm` - Choose from DQN, PPO, SAC or A2C
* `--time_steps` - Number of training steps

### Action space transfer
Sim-to-sim example:  
`python3 doom_baselines_agents.py --mode train --algorithm DQN --environment sim-to-sim --source_model "./example_models/Doom_source_model/DQN/DQN_source_model_1.pkl" --time_steps 20000`

Turtlebot example:  
`python3 doom_baselines_agents.py --mode train --algorithm DQN --environment turtlebot --source_model "./example_models/Doom_source_model/DQN/DQN_source_model_1.pkl" --time_steps 20000 --visualize 1`

Some optional arguments for action space transfer (Supported only for DQN and PPO):
* `--replace` - Used for replace method. Set to 0 for adapter method.
* `--source_model` - Defines the source from which the weights are loaded.
* `--restore_value_function` - Restores parameters from previous network value function.
* `--freeze_base_nn` - Sets networks parameters non-trainable (except output layer).
* `--freeze_vf` - Sets value function parameters non-trainable.
* `--load_params_from_previous_model` - Loads parameters from source model.  
Check default values from the code.

## Testing
Test the saved agent. Remember to specify environment, so that it matches the environment where agent was trained. Use `--visualize 1` argument to show the gameplay. Note that due to frameskip, the gameplay is shown at ~3 fps.

Example:  
`python3 doom_baselines_agents.py --mode test --algorithm DQN --environment doom --model [path to model]`

## Test logs
Example to produce the test logs:  
`python3 eval_doom_baselines.py [directory of checkpoints] --algorithm PPO --environment Doom --action_space discrete --recursive`

## Plotting
Training logs can be plotted by placing the "log/" directory inside "./plots/". Mean of multiple plots can be plotted by adding multiple directories. Example of one possible directory structure is provided in the "./plots/" directory. You can plot the example logs with:  
 
`python3 plot_from_logs.py --plot_style multiple_means`

Some of the possible arguments:
* `--plot_style` - Choose from 'mean', 'single', and 'multiple_means'
* `--smooth_algo` - Two different options for smoothing: 'fixed' and 'interp'
* `--save` - Saves the plot as pdf
* `--mode` - Plot 'reward' or 'episode_length'

## Changes to stable baselines library:
Stable baselines is originally from: https://github.com/hill-a/stable-baselines.
Some modifications were made to support the action space transfer:
* DQN and PPO2 can take in parameters `freeze_base_nn`, `freeze_vf` and `freeze_last_fc`
* An parameter `adapter_old_action_space` can be given for classes `FeedForwardPolicy(ActorCriticPolicy)` from /common/policies.py and `FeedForwardPolicy(DQNPolicy)` from /deepq/policies.py, to support adapter method in action space transfer.