import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import A2C, PPO, DQN

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image, ImageDraw
import io

register(
    id='2048-eval',
    entry_point='envs:Eval2048Env'
)

def get_color_palette():
    colors = [
        "#eee4da",  # 2
        "#ede0c8",  # 4
        "#f2b179",  # 8
        "#f59563",  # 16
        "#f67c5f",  # 32
        "#f65e3b",  # 64
        "#edcf72",  # 128
        "#edcc61",  # 256
        "#edc850",  # 512
        "#edc53f",  # 1024
        "#edc22e"   # 2048
    ]
    return sns.color_palette(colors)

# Function to visualize a single matrix and return it as an in-memory image
def matrix_to_image(matrix):
    matrix_float = matrix.astype(float)
    norm_matrix = np.log2(matrix_float, where=matrix_float > 0, out=np.zeros_like(matrix_float))
    norm_matrix = np.clip(norm_matrix, 0, len(get_color_palette()) - 1)

    plt.figure(figsize=(6, 6))
    ax = sns.heatmap(norm_matrix, annot=matrix, cmap=get_color_palette(), linewidths=2, linecolor='white', cbar=False, square=True, fmt="g")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("")

    # Save plot to an in-memory file
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    # Return the in-memory image
    return Image.open(buf)

def write_gif(env, model):
    done = False
    # Set seed and reset env using Gymnasium API
    obs, info = env.reset(seed=0)

    matrices = []

    while not done:
        # Interact with env using Gymnasium API
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        matrices.append(np.array(env.Matrix))

    # Generate images in memory and convert them into a GIF
    images = [matrix_to_image(matrix) for matrix in matrices]
    images[0].save('matrix_animation.gif', save_all=True, append_images=images[1:], duration=500, loop=0)

    print("GIF created as 'matrix_animation.gif'")

def main():
    model_path = "models/sample_model/0"  # Change path name to load different models
    env = gym.make('2048-eval')

    model = PPO.load(model_path)
    write_gif(env, model)

if __name__ == "__main__":
    main()
