import numpy as np
import torch
import torch.nn.functional as F

from .wrappers import LazyFrames

def states_preprocessor(states, device='cpu', dtype=torch.float32):
    """
    Convert list of states into the form suitable for model.
    :param states: list of numpy arrays or LazyFrames
    :param device: target device for tensor (e.g., 'cpu' or 'cuda')
    :param dtype: data type of the tensor (e.g., torch.float32)
    :return: torch.Tensor on specified device
    """
    processed_states = []
    for state in states:
        if isinstance(state, LazyFrames):
            state = np.array(state)
        if isinstance(state, np.ndarray):
            if state.ndim == 3:
                state = np.expand_dims(state, 0)
            processed_states.append(state)
        else:
            raise ValueError(f"Unexpected state type: {type(state)}. Expected LazyFrames or np.ndarray.")
    np_states = np.concatenate(processed_states, axis=0)
    return torch.tensor(np_states, device=device, dtype=dtype)


class ProbabilityActionSelector():
    def __call__(self, probs):
        assert isinstance(probs, np.ndarray)
        actions = []
        for prob in probs:
            actions.append(np.random.choice(len(prob), p=prob))
        return np.array(actions)


class PolicyAgent:
    def __init__(self, model, action_selector=ProbabilityActionSelector(),
                 device="cpu", apply_softmax=False,
                 preprocessor=states_preprocessor):
        self.model = model
        self.action_selector = action_selector
        self.device = device
        self.apply_softmax = apply_softmax
        self.preprocessor = preprocessor

    @torch.no_grad()
    def __call__(self, states, agent_states=None):
        if agent_states is None:
            agent_states = [None] * len(states)
        if self.preprocessor is not None:
            states = self.preprocessor(states, device=self.device)
        prob_v = self.model(states)
        if self.apply_softmax:
            prob_v = F.softmax(prob_v, dim=1)
        probs = prob_v.data.cpu().numpy()
        actions = self.action_selector(probs)
        return np.array(actions), agent_states
