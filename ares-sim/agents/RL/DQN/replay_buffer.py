import random
from collections import deque
import torch


class ReplayBuffer:
    """Stores pre-encoded transitions as tensors.

    All data is stored as tensors ready for batched training.
    The caller is responsible for encoding observations before pushing.
    """

    def __init__(self, capacity: int):
        self.memory: deque[tuple[torch.Tensor, ...]] = deque(maxlen=capacity)

    def push(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ):
        """Store a single transition.

        Args:
            state: Encoded observation tensor, shape (1, state_dim) or (state_dim,).
            action: Fixed-size action tensor, shape (num_zones,). Each position
                    corresponds to one zone; value is the action index taken.
            reward: Scalar reward for this transition.
            next_state: Encoded next observation tensor, same shape as state.
            done: Whether the episode ended after this transition.
        """
        self.memory.append((
            state.detach().squeeze(0),                          # (state_dim,)
            action.detach(),                                    # (num_zones,)
            torch.tensor(reward, dtype=torch.float32),          # scalar
            next_state.detach().squeeze(0),                     # (state_dim,)
            torch.tensor(float(done), dtype=torch.float32),     # scalar
        ))

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a batch and return pre-stacked tensors.

        Returns:
            states:      (batch_size, state_dim)
            actions:     (batch_size, num_zones)
            rewards:     (batch_size,)
            next_states: (batch_size, state_dim)
            dones:       (batch_size,)
        """
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.stack(states),
            torch.stack(actions),
            torch.stack(rewards),
            torch.stack(next_states),
            torch.stack(dones),
        )

    def __len__(self):
        return len(self.memory)
