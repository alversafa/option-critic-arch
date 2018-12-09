import numpy as np

# logsumexp() and expit() are used because they are
# numerically stable
# expit() is the sigmoid function
from scipy.misc import logsumexp
from scipy.special import expit


class EpsGreedyPolicy():

	def __init__(self, rng, nstates, noptions, epsilon):
		self.rng = rng
		self.nstates = nstates
		self.noptions = noptions
		self.epsilon = epsilon
		self.Q_Omega_table = np.zeros((nstates, noptions))

	def Q_Omega(self, state, option=None):
		if option is None:
			return self.Q_Omega_table[state,:]
		else:
			return self.Q_Omega_table[state, option]

	def sample(self, state):
		if self.rng.uniform() < self.epsilon:
			return int(self.rng.randint(self.noptions))
		else:
			return int(np.argmax(self.Q_Omega(state)))


class SoftmaxPolicy():

	def __init__(self, rng, lr, nstates, nactions, temperature=1.0):
		self.rng = rng
		self.lr = lr
		self.nstates = nstates
		self.nactions = nactions
		self.temperature = temperature
		self.weights = np.zeros((nstates, nactions))

	def Q_U(self, state, action=None):
		if action is None:
			return self.weights[state,:]
		else:
			return self.weights[state, action]

	def pmf(self, state):
		exponent = self.Q_U(state) / self.temperature
		return np.exp(exponent - logsumexp(exponent))

	def sample(self, state):
		return int(self.rng.choice(self.nactions, p=self.pmf(state)))

	def gradient(self):
		pass

	def update(self, state, action, Q_U):
		actions_pmf = self.pmf(state)
		self.weights[state, :] -= self.lr * actions_pmf * Q_U
		self.weights[state, action] += self.lr * Q_U


class SigmoidTermination():

	def __init__(self, rng, lr, nstates):
		self.rng = rng
		self.lr = lr
		self.nstates = nstates
		self.weights = np.zeros((nstates,))

	def pmf(self, state):
		return expit(self.weights[state])

	def sample(self, state):
		return int(self.rng.uniform() < self.pmf(state))

	def gradient(self, state):
		return self.pmf(state) * (1.0 - self.pmf(state)), state

	def update(self, state, advantage):
		magnitude, direction = self.gradient(state)
		self.weights[direction] -= self.lr * magnitude * advantage


class Critic():

	def __init__(self, lr, discount, Q_Omega_table, nstates, noptions, nactions):
		self.lr = lr
		self.discount = discount
		self.Q_Omega_table = Q_Omega_table
		self.Q_U_table = np.zeros((nstates, noptions, nactions))

	def cache(self, state, option, action):
		self.last_state = state
		self.last_option = option
		self.last_action = action
		self.last_Q_Omega = self.Q_Omega(state, option)

	def Q_Omega(self, state, option=None):
		if option is None:
			return self.Q_Omega_table[state, :]
		else:
			return self.Q_Omega_table[state, option]

	def Q_U(self, state, option, action):
		return self.Q_U_table[state, option, action]

	def A_Omega(self, state, option=None):
		advantage = self.Q_Omega(state) - np.max(self.Q_Omega(state))

		if option is None:
			return advantage
		else:
			return advantage[option]

	def update_Qs(self, state, option, action, reward, done, terminations):
		# One step target for Q_Omega
		target = reward
		if not done:
			beta_omega = terminations[self.last_option].pmf(state)
			target += self.discount * ((1.0 - beta_omega)*self.Q_Omega(state, self.last_option) + \
						beta_omega*np.max(self.Q_Omega(state)))

		# Difference update
		tderror_Q_Omega = target - self.last_Q_Omega
		self.Q_Omega_table[self.last_state, self.last_option] += self.lr * tderror_Q_Omega

		tderror_Q_U = target - self.Q_U(self.last_state, self.last_option, self.last_action)
		self.Q_U_table[self.last_state, self.last_option, self.last_action] += self.lr * tderror_Q_U

		# Cache
		self.last_state = state
		self.last_option = option
		self.last_action = action
		if not done:
			self.last_Q_Omega = self.Q_Omega(state, option)
