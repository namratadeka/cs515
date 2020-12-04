import torch
from torch import nn
import numpy as np

from push_policy.models.network import Network


class Actor(nn.Module):
	def __init__(self, cfg):
		super(Actor, self).__init__()
		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		self.state_fc = Network(cfg["state_fc"])
		self.cnn = Network(cfg["cnn"])
		self.actor_fc = Network(cfg["actor_fc"])

	def forward(self, state, image):
		if isinstance(state, np.ndarray):
			state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
		if isinstance(image, np.ndarray):
			image = torch.tensor(image, dtype=torch.float).permute(2, 0, 1).unsqueeze(0).to(self.device)

		state_features = self.state_fc(state)
		img_features = self.cnn(image)
		img_features = torch.flatten(img_features, start_dim=1)
		features = torch.cat([state_features, img_features], dim=1)
		action = self.actor_fc(features)

		return action

class Critic(nn.Module):
	def __init__(self, cfg):
		super(Critic, self).__init__()
		self.state_fc = Network(cfg["state_fc"])
		self.cnn = Network(cfg["cnn"])
		self.critic_fc = Network(cfg["critic_fc"])

	def forward(self, state, image):
		if isinstance(state, np.ndarray):
			state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
		if isinstance(image, np.ndarray):
			image = torch.tensor(image, dtype=torch.float).permute(2, 0, 1).unsqueeze(0).to(self.device)

		state_features = self.state_fc(state)
		img_features = self.cnn(image)
		img_features = torch.flatten(img_features, start_dim=1)
		features = torch.cat([state_features, img_features], dim=1)
		value = self.critic_fc(features)

		return value
