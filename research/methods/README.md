# Research Method Implementations

This directory contains real implementations of paper-inspired long-horizon RL agents.

## Implemented Agents

### HCAPO (Hierarchical Constrained Actor-critic with Planning Optimization)
- **File**: `hcapo_agent.py`
- **Features**:
  - Hierarchical policy with high-level planner and low-level executor
  - Hindsight Experience Replay (HER) with goal relabeling
  - Constraint-aware action selection
  - Subgoal decomposition based on enrollment milestones
- **Training**: Real policy gradient updates with hindsight relabeling

### MiRA (Milestone-based Reward Augmentation)
- **File**: `mira_agent.py`
- **Features**:
  - Learned potential function for reward shaping
  - Potential-based reward augmentation: F(s,s') = γΦ(s') - Φ(s)
  - Milestone achievement tracking and bonus
  - TD-learning for potential critic
- **Training**: Joint policy and potential critic updates

### KLong (Long-context Trajectory Processing)
- **File**: `klong_agent.py`
- **Features**:
  - Multi-scale temporal abstraction (1, 5, 20, 60 step windows)
  - TD(λ) with eligibility traces
  - Trajectory segmentation with overlap
  - Context-aware policy and value functions
- **Training**: Segment-wise policy gradient with eligibility trace updates

### MemexRL (Memory-Augmented RL)
- **File**: `memex_agent.py`
- **Features**:
  - Learned memory write gate (decides when to store)
  - Attention-based memory retrieval
  - Memory importance scoring with hindsight
  - Memory-augmented policy and value function
- **Training**: Joint policy, write gate, and importance network updates

## Training Results

Run training with:
```bash
python experiments/train_agents.py --agent all --episodes 50
```

Latest results (50 episodes each):
- HCAPO: 0.1746 average score
- MiRA: 0.1817 average score
- KLong: 0.2064 average score
- MemexRL: 0.2518 average score

Trained models saved to `data/trained_agents/`.
Training curves saved to `data/training_results/`.

## Architecture

All agents share a common neural network infrastructure:
- `training/neural_policy.py`: NeuralNetwork, ActorCritic base classes
- Feature extraction: 37-dimensional state vector from observations
- Gradient clipping for stability
- Xavier weight initialization

## Tests

```bash
python test_agents.py
```

Verifies:
- Neural network forward/backward passes
- Actor-critic action selection and value estimation
- Agent-specific features (hindsight, potential, segmentation, memory)
- Save/load serialization
