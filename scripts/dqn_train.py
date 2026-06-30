import torch
import torch.optim as optim
from core.state import Side, CommanderMemory
from core.obs import build_obs
from tick import TickEngine  # Assuming TickEngine is stored here
from config.seeds import get_seed_1 [cite: 27]
from agents.RL.DQN.DQNCommander import DQNCommander
from agents.RL.DQN.trainer import Trainer
from agents.RL.DQN.network import DQNNetwork

def run_training():
    # 1. Settings Setup
    state_dim = 11  # Based on ObservationEncoder feature size [cite: 17]
    hidden_dim = 128
    batch_size = 32
    gamma = 0.99
    episodes = 500
    checkpoint_interval = 50

    # 2. Build Networks and Trainer
    online_net = DQNNetwork(state_dim, hidden_dim) [cite: 2, 18]
    target_net = DQNNetwork(state_dim, hidden_dim)
    target_net.load_state_dict(online_net.state_dict())
    
    optimizer = optim.Adam(online_net.parameters(), lr=1e-3)
    
    # We share a temporary buffer placeholder for self-play training setups
    from agents.RL.DQN.replay_buffer import ReplayBuffer
    shared_buffer = ReplayBuffer(capacity=10000) [cite: 3]

    trainer = Trainer(
        online_net=online_net,
        target_net=target_net,
        optimizer=optimizer,
        replay_buffer=shared_buffer,
        batch_size=batch_size,
        gamma=gamma,
        sync_every_k_steps=100
    ) [cite: 22]

    # 3. Create Commanders
    blue_agent = DQNCommander(state_dim, hidden_dim, trainer=trainer)
    red_agent = DQNCommander(state_dim, hidden_dim, trainer=trainer)
    
    # Override independent buffers with our single collection target
    blue_agent.replay_buffer = shared_buffer
    red_agent.replay_buffer = shared_buffer

    # 4. Training Loop
    for episode in range(1, episodes + 1):
        initial_state = get_seed_1() [cite: 27]
        engine = TickEngine(initial_state, blue_commander=blue_agent, red_commander=red_agent)
        
        # Run episode until completion
        engine.run()

        # Step optimization after each complete match episode
        loss_val = blue_agent.train_step()
        
        if episode % 10 == 0:
            current_eps = blue_agent.epsilon_scheduler.get_epsilon(blue_agent.tick_counter) [cite: 4]
            print(f"Episode {episode} | Loss: {loss_val} | Epsilon: {current_eps:.3f}")

        # Save model periodic weights
        if episode % checkpoint_interval == 0:
            blue_agent.save(f"checkpoint_dqn_ep_{episode}.pt")
            print(f"Saved Checkpoint for episode {episode}.")

if __name__ == "__main__":
    run_training()