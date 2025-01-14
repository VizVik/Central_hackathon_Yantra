# Step 1: Import Libraries
import joblib
import gym
from gym import spaces
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Step 2: Load Dataset
crop_data = pd.read_csv('crop_yield.csv') 

# Step 3: Encode Categorical Variables
label_encoders = {}
for column in ['Region', 'Soil_Type', 'Weather_Condition', 'Crop']:
    le = LabelEncoder()
    crop_data[column] = le.fit_transform(crop_data[column])
    label_encoders[column] = le

# Step 4: Scale Numerical Features
scaler = StandardScaler()

# Define feature columns (exclude the target 'Crop')
feature_columns = crop_data.columns.difference(['Crop'])

# Scale only the feature columns
crop_data[feature_columns] = scaler.fit_transform(crop_data[feature_columns])

# Step 5: Save Metadata
joblib.dump(label_encoders, "label_encoders.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(feature_columns, "feature_columns.pkl")

# Step 6: Define Custom Gym Environment
class CropRotationEnv(gym.Env):
    def __init__(self, data):
        super(CropRotationEnv, self).__init__()

        # Dataset
        self.data = data
        self.num_samples = len(data)
        
        # Action Space: Crop Choices (unique crops as discrete actions)
        self.crop_types = list(data['Crop'].unique())
        self.action_space = spaces.Discrete(len(self.crop_types))
        
        # Observation Space: State features (scaled)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(data.columns) - 1,), dtype=np.float32
        )
        
        # Initialize state and index
        self.current_step = 0
        self.current_state = None
        
    def reset(self):
        """Reset the environment to a random initial state."""
        self.current_step = np.random.randint(0, self.num_samples)
        self.current_state = self.data.iloc[self.current_step, :-1].values.astype(np.float32)
        return self.current_state

    def step(self, action):
        """Take an action and return the next state, reward, done, info."""
        chosen_crop = self.crop_types[action]
        actual_crop = self.data.iloc[self.current_step]['Crop']
        
        # Calculate reward: positive for matching crops, penalize mismatch
        reward = 1 if chosen_crop == actual_crop else -0.5
        
        # Simulate next step
        self.current_step = (self.current_step + 1) % self.num_samples
        next_state = self.data.iloc[self.current_step, :-1].values.astype(np.float32)
        done = self.current_step == self.num_samples - 1
        
        return next_state, reward, done, {}
    
    def render(self, mode='human'):
        print(f"Step: {self.current_step}, State: {self.current_state}")
    
    def close(self):
        pass

# Step 7: Create the Environment
env = CropRotationEnv(crop_data)

# Step 8: Train the RL Model
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# Step 9: Save the Model
model.save("crop_rotation_model")
print("Training completed and model saved.")