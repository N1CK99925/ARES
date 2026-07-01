import torch


class Trainer:
    """DQN trainer that operates purely on tensors.

    Knows only about the online network, target network, optimizer,
    replay buffer, discount factor, and batch size. Does not import
    or manipulate commanders, observations, or simulation classes.
    """

    def __init__(
        self,
        online_net,
        target_net,
        optimizer,
        replay_buffer,
        batch_size,
        gamma,
        sync_every_k_steps,
        max_grad_norm=10.0,
    ):
        self.online_net = online_net
        self.target_net = target_net
        self.optimizer = optimizer
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.gamma = gamma
        self.loss_fn = torch.nn.SmoothL1Loss()
        self.sync_every_k_steps = sync_every_k_steps
        self.max_grad_norm = max_grad_norm
        self.step_count = 0

    def train_step(self):
        """Run one optimization step on a batch sampled from replay.

        Returns the loss value, or None if not enough experience yet.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None  # not enough experience yet, skip

        # Replay buffer returns pre-stacked tensors
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = (
            self.replay_buffer.sample(self.batch_size)
        )
        # obs_batch:      (batch_size, state_dim)
        # action_batch:   (batch_size, 5)        — one action index per zone
        # reward_batch:   (batch_size,)
        # next_obs_batch: (batch_size, state_dim)
        # done_batch:     (batch_size,)

        online_outputs = self.online_net(obs_batch)        # dict: zone_1..zone_5 -> (batch, out_size_z)
        with torch.no_grad():
            online_next_outputs = self.online_net(next_obs_batch)
            target_next_outputs = self.target_net(next_obs_batch)

        total_loss = torch.tensor(0.0)
        zone_losses = {}
        zone_q_stats = {}
        for zone_idx, zone_id in enumerate(range(1, 6)):
            zone_key = f"zone_{zone_id}"

            # Online Q-values for this zone, shape (batch_size, out_size_z)
            q_online_z = online_outputs[zone_key]

            # Gather the Q-value of the action actually taken for this zone
            q_taken_z = torch.gather(q_online_z, 1, action_batch[:, zone_idx].unsqueeze(1)).squeeze(1)

            # Double DQN: online net selects action, target net evaluates it
            best_actions = online_next_outputs[zone_key].argmax(dim=1, keepdim=True)
            q_next_z = target_next_outputs[zone_key].gather(1, best_actions).squeeze(1)

            # TD target: reward + gamma * Q_target(s', argmax_a Q_online(s', a)) * (1 - done)
            target_z = (reward_batch + self.gamma * q_next_z * (1 - done_batch)).detach()

            loss_z = self.loss_fn(q_taken_z, target_z)
            total_loss = total_loss + loss_z
            zone_losses[zone_id] = loss_z.item()
            zone_q_stats[zone_id] = {
                "mean": q_online_z.mean().item(),
                "max": q_online_z.max().item(),
            }

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=self.max_grad_norm)
        self.optimizer.step()
        self.step_count += 1
        if self.step_count % self.sync_every_k_steps == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
        return total_loss.item(), zone_losses, zone_q_stats
