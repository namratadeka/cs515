import torch
from torch import nn

from models.network import Network


class Actor(nn.Module):
	def __init__(self, cfg):
		self.state_fc = Network(cfg["state_fc"])
		self.cnn = Network(cfg["cnn"])
		self.actor_fc = Network(cfg["actor_fc"])

	def forward(self, state, image):
		state_features = self.state_fc(state)
		img_features = self.cnn(image)
		img_features = torch.flatten(img_features, start_dim=1)
		features = torch.concat([state_features, img_features], dim=0)
		action = self.actor_fc(features)

		return action

class Critic(nn.Module):
	def __init__(self, cfg):
		self.state_fc = Network(cfg["state_fc"])
		self.cnn = Network(cfg["cnn"])
		self.critic_fc = Network(cfg["critic_fc"])

	def forward(self, state, image):
		state_features = self.state_fc(state)
		img_features = self.cnn(image)
		img_features = torch.flatten(img_features, start_dim=1)
		features = torch.concat([state_features, img_features], dim=0)
		value = self.critic_fc(features)

		return value
