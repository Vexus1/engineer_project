import numpy as np
import torch
import torch.nn.functional as F

def states_preprocessor(states, device='cpu', dtype=torch.float32):
    """
    Convert list of states into the form suitable for model.
    :param states: list of numpy arrays or a single numpy array
    :param device: target device for tensor (e.g., 'cpu' or 'cuda')
    :param dtype: data type of the tensor (e.g., torch.float32)
    :return: torch.Tensor on specified device
    """
    if isinstance(states, np.ndarray):
        np_states = states if states.ndim > 1 else np.expand_dims(states, 0)
    else:
        if len(states) == 1:
            np_states = np.expand_dims(states[0], 0)
        else:
            np_states = np.array([np.array(s, copy=False) for s in states],
                                  copy=False)
    return torch.tensor(np_states, device=device, dtype=dtype)


class ProbabilityActionSelector():
    def __call__(self, probs):
        assert isinstance(self, np.array)
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
            states = self.preprocessor(states)
            if torch.is_tensor(states):
                states = states.to(self.device)
        prob_v = self.model(states)
        if self.apply_softmax:
            prob_v = F.softmax(prob_v, dim=1)
        probs = prob_v.data.cpu().numpy()
        actions = self.action_selector(probs)
        return np.array(actions), agent_states  
