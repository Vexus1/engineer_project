import numpy as np
import torch
import torch.nn.functional as F

def states_preprocessor(states):
    """
    Convert list of states into the form suitable for model.
    :param states: list of numpy arrays or LazyFrames
    :param device: target device for tensor (e.g., 'cpu' or 'cuda')
    :param dtype: data type of the tensor (e.g., torch.float32)
    :return: torch.Tensor on specified device
    """
    if len(states) == 1:
        np_states = np.expand_dims(states[0], 0)
    else:
        np_states = np.asarray([np.asarray(s) for s in states])
    return torch.tensor(np_states)


class ProbabilityActionSelector():
    def __call__(self, probs):
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
        probs_v = self.model(states)
        if self.apply_softmax:
            probs_v = F.softmax(probs_v, dim=1)
        probs = probs_v.data.cpu().numpy()
        actions = self.action_selector(probs)
        return np.array(actions), agent_states