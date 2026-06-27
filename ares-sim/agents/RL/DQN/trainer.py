import torch

class Trainer:
    def __init__(self, online_net, target_net, optimizer, replay_buffer, batch_size, gamma, sync_every_k_steps):
        self.online_net = online_net
        self.target_net = target_net
        self.optimizer = optimizer
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.gamma = gamma
        self.loss_fn = torch.nn.MSELoss()
        self.sync_every_k_steps = sync_every_k_steps
        self.step_count = 0

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return None  # not enough experience yet, skip

        batch = self.replay_buffer.sample(self.batch_size)
        # batch is a list of (obs, action, reward, next_obs, done) tuples 
        
        # TODO 1: unzip the batch into separate tensors
        obs_list, action_list, reward_list, next_obs_list, done_list = zip(*batch)
        obs_batch = torch.cat(obs_list, dim=0)          # (batch_size, state_dim)
        action_batch = torch.tensor(action_list)         # (batch_size, 5)
        reward_batch = torch.tensor(reward_list, dtype=torch.float32)
        next_obs_batch = torch.cat(next_obs_list, dim=0)
        done_batch = torch.tensor(done_list, dtype=torch.float32)

        online_outputs = self.online_net(obs_batch)        # dict: zone_1..zone_5 -> (batch, out_size_z)
        with torch.no_grad():
            target_outputs = self.target_net(next_obs_batch)

        total_loss = torch.tensor(0.0)
        for zone_idx, zone_id in enumerate(range(1, 6)):
            zone_key = f"zone_{zone_id}"

            # TODO 2: get this zone's online Q-values for the WHOLE batch
            #   shape (batch_size, out_size_z)
            q_online_z = online_outputs[zone_key]

            # TODO 3: gather the Q-value of the ACTION ACTUALLY TAKEN for this zone
            #   action_batch[:, zone_idx] gives you the idx per sample
            #   look up torch.gather — this is the standard tool for "pick one value per row"

            q_taken_z = torch.gather(q_online_z, 1, action_batch[:, zone_idx].unsqueeze(1)).squeeze(1)


            # TODO 4: get this zone's TARGET net Q-values for next_obs, take max per row
            q_next_max_z = target_outputs[zone_key].max(dim=1).values

            # TODO 5: compute TD target
            # target = reward + gamma * q_next_max_z * (1 - done)
           #   note (1 - done): if episode ended, there IS no next state, zero out the future term
            target_z = (reward_batch + self.gamma * q_next_max_z * (1 - done_batch)).detach()

            # TODO 6: MSE loss between q_taken_z and target_z (detach target_z from graph)
            loss_z = self.loss_fn(q_taken_z, target_z)
            total_loss = total_loss + loss_z
            
           

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        self.step_count += 1
        if self.step_count % self.sync_every_k_steps == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
        return total_loss.item()
