import gymnasium as gym
import ale_py

# Rejestracja środowisk Atari
gym.register_envs(ale_py)

# Inicjalizacja środowiska dla Pong
env = gym.make("ALE/Pong-v5")

# Resetowanie środowiska w celu wygenerowania pierwszej obserwacji
observation, info = env.reset(seed=42)

for _ in range(1000):
    # Przykładowa polityka (losowy wybór akcji)
    action = env.action_space.sample()

    # Wykonanie kroku w środowisku
    observation, reward, terminated, truncated, info = env.step(action)

    # Jeśli epizod zakończył się, resetujemy środowisko
    if terminated or truncated:
        observation, info = env.reset()

# Zamknięcie środowiska
env.close()
