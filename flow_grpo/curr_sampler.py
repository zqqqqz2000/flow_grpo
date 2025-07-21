import math
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from typing import Iterator, Optional, List, Dict, Any
from collections import defaultdict


class CurrDistributedSampler(DistributedSampler):
    """
    Distributed sampler supporting four curriculum learning strategies:

    - timestep: Fixed order based on difficulty (easy to hard)
    - balance: Random uniform sampling across all difficulties
    - cosine: Smooth transition from easy to hard using cosine schedule
    - gaussian: Bell-curve distribution with adjustable parameters
    """

    def __init__(
        self,
        dataset,
        strategy: str = "balance",
        total_steps: int = 300,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 0,
        drop_last: bool = False,
        alpha: float = 1.0,  # Gaussian width parameter
        beta: float = 1.0,  # Gaussian progression speed parameter
    ):
        """
        Initialize curriculum distributed sampler.

        Args:
            dataset: Dataset to sample from
            strategy: Sampling strategy ("timestep", "balance", "cosine", "gaussian")
            total_steps: Total steps needed for cosine and gaussian sampler
            num_replicas: Number of processes participating in distributed training
            rank: Rank of current process within distributed training
            seed: Random seed for reproducibility
            drop_last: Whether to drop the last incomplete batch
            alpha: Gaussian schedule width parameter (larger = wider difficulty spread)
            beta: Gaussian schedule progression speed (larger = faster progression to hard samples)
        """
        super().__init__(dataset=dataset, num_replicas=num_replicas, rank=rank, seed=seed, shuffle=False)

        if strategy not in ["timestep", "balance", "cosine", "gaussian"]:
            raise ValueError(f"Invalid strategy: {strategy}. Must be one of: timestep, balance, cosine, gaussian")

        self.strategy = strategy
        self.alpha = alpha
        self.beta = beta

        # Build difficulty mapping from dataset
        self._build_difficulty_mapping()

        self.total_steps = total_steps
        self.current_epoch = 0

    def _build_difficulty_mapping(self) -> None:
        """Build mapping from difficulty levels to sample indices."""
        self.difficulty_to_indices: Dict[Any, List[int]] = defaultdict(list)

        for idx in range(len(self.dataset)):
            try:
                sample = self.dataset[idx]
                if isinstance(sample, dict):
                    difficulty = sample.get("difficulty", 0)
                elif hasattr(sample, "difficulty"):
                    difficulty = sample.difficulty
                else:
                    difficulty = 0
            except:
                difficulty = 0

            self.difficulty_to_indices[difficulty].append(idx)

        # Sort difficulties and ensure at least one exists
        self.unique_difficulties = sorted(self.difficulty_to_indices.keys())
        self.num_difficulties = len(self.unique_difficulties)

        if self.num_difficulties == 0:
            self.unique_difficulties = [0]
            self.difficulty_to_indices[0] = list(range(len(self.dataset)))
            self.num_difficulties = 1

    def _get_timestep_indices(self) -> List[int]:
        """Return indices in fixed difficulty order (easy to hard)."""
        indices = []
        for difficulty in self.unique_difficulties:
            indices.extend(self.difficulty_to_indices[difficulty])
        return indices

    def _get_balance_indices(self) -> List[int]:
        """Return randomly shuffled indices (uniform sampling)."""
        indices = list(range(len(self.dataset)))
        g = torch.Generator()
        g.manual_seed(self.seed + self.current_epoch)
        perm = torch.randperm(len(indices), generator=g)
        return [indices[i] for i in perm]

    def _get_cosine_indices(self) -> List[int]:
        """Return indices using cosine curriculum schedule."""
        # Calculate progress through training (0 to 1)
        progress = min(self.current_epoch / max(self.total_steps - 1, 1), 1.0)

        # Cosine weight: 1.0 (easy focus) at start, 0.0 (hard focus) at end
        cosine_weight = 0.5 * (1.0 + math.cos(math.pi * progress))

        # Calculate sampling probabilities for each difficulty
        difficulty_probs = []
        for i in range(self.num_difficulties):
            # Normalize difficulty position to [0, 1]
            normalized_pos = i / max(self.num_difficulties - 1, 1)
            # Higher weight for easier difficulties early in training
            prob = cosine_weight * (1.0 - normalized_pos) + (1.0 - cosine_weight) * normalized_pos
            prob = max(prob, 0.05)  # Minimum probability to ensure some diversity
            difficulty_probs.append(prob)

        return self._sample_by_difficulty_probs(difficulty_probs)

    def _get_gaussian_indices(self) -> List[int]:
        """Return indices using Gaussian curriculum schedule."""
        # Calculate progress through training (0 to 1)
        progress = min(self.current_epoch / max(self.total_steps - 1, 1), 1.0)

        # Center of Gaussian moves from easy (0) to hard (num_difficulties-1)
        center = progress * (self.num_difficulties - 1) * self.beta

        # Calculate Gaussian probabilities for each difficulty
        difficulty_probs = []
        for i in range(self.num_difficulties):
            # Gaussian centered at 'center' with spread controlled by alpha
            prob = math.exp(-0.5 * ((i - center) / max(self.alpha, 0.1)) ** 2)
            difficulty_probs.append(prob)

        return self._sample_by_difficulty_probs(difficulty_probs)

    def _sample_by_difficulty_probs(self, difficulty_probs: List[float]) -> List[int]:
        """Sample dataset indices based on difficulty probabilities."""
        # Normalize probabilities
        total_prob = sum(difficulty_probs)
        if total_prob <= 0:
            difficulty_probs = [1.0] * len(difficulty_probs)
            total_prob = len(difficulty_probs)

        normalized_probs = [p / total_prob for p in difficulty_probs]

        # Calculate number of samples to draw from each difficulty
        samples_per_difficulty = []
        total_samples = len(self.dataset)

        for i, prob in enumerate(normalized_probs):
            if i == len(normalized_probs) - 1:
                # Last difficulty gets remaining samples
                remaining = total_samples - sum(samples_per_difficulty)
                samples_per_difficulty.append(max(0, remaining))
            else:
                num_samples = int(total_samples * prob)
                samples_per_difficulty.append(num_samples)

        # Sample indices from each difficulty level
        indices = []
        g = torch.Generator()
        g.manual_seed(self.seed + self.current_epoch)

        for difficulty, num_samples in zip(self.unique_difficulties, samples_per_difficulty):
            difficulty_indices = self.difficulty_to_indices[difficulty]

            if num_samples <= 0:
                continue
            elif num_samples >= len(difficulty_indices):
                # Use all indices from this difficulty (with repetition if needed)
                repeat_count = num_samples // len(difficulty_indices)
                remainder = num_samples % len(difficulty_indices)

                indices.extend(difficulty_indices * repeat_count)
                if remainder > 0:
                    perm = torch.randperm(len(difficulty_indices), generator=g)[:remainder]
                    indices.extend([difficulty_indices[i] for i in perm])
            else:
                # Randomly sample from this difficulty
                perm = torch.randperm(len(difficulty_indices), generator=g)[:num_samples]
                indices.extend([difficulty_indices[i] for i in perm])

        # Final shuffle of all selected indices
        if indices:
            perm = torch.randperm(len(indices), generator=g)
            indices = [indices[i] for i in perm]

        return indices

    def __iter__(self) -> Iterator[int]:
        """Generate indices for current epoch."""
        # Select sampling strategy
        if self.strategy == "timestep":
            indices = self._get_timestep_indices()
        elif self.strategy == "balance":
            indices = self._get_balance_indices()
        elif self.strategy == "cosine":
            indices = self._get_cosine_indices()
        elif self.strategy == "gaussian":
            indices = self._get_gaussian_indices()
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        # Ensure we have enough indices for distributed training
        if not self.drop_last:
            # Pad with repetition to match total_size
            padding_size = self.total_size - len(indices)
            if padding_size > 0:
                if padding_size <= len(indices):
                    indices += indices[:padding_size]
                else:
                    # Repeat indices multiple times if needed
                    repetitions = math.ceil(padding_size / len(indices))
                    extended_indices = (indices * repetitions)[:padding_size]
                    indices += extended_indices
        else:
            # Truncate to match total_size
            indices = indices[: self.total_size]

        assert len(indices) == self.total_size, f"Expected {self.total_size} indices, got {len(indices)}"

        # Subsample for this specific process/rank
        process_indices = indices[self.rank : self.total_size : self.num_replicas]
        assert (
            len(process_indices) == self.num_samples
        ), f"Expected {self.num_samples} indices for rank {self.rank}, got {len(process_indices)}"

        return iter(process_indices)

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for this sampler to ensure different shuffling per epoch."""
        super().set_epoch(epoch)
        self.current_epoch = epoch

    def __len__(self) -> int:
        """Return the number of samples for this process."""
        return self.num_samples
