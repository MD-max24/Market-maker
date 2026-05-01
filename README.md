# Market-maker
# 🤖 Market Making Optimal avec Q-Learning

> Projet pédagogique de reinforcement learning appliqué à la finance de marché.  
> Un agent apprend à placer des ordres bid/ask sur un carnet d'ordres simulé, en tenant compte des coûts de transaction et de la toxicité du flow.

---

## 📋 Table des matières

- [Contexte et motivation](#contexte-et-motivation)
- [Architecture du projet](#architecture-du-projet)
- [Modélisation](#modélisation)
- [Structure des fichiers](#structure-des-fichiers)
- [Utilisation](#utilisation)
- [Paramètres](#paramètres)
- [Résultats attendus](#résultats-attendus)
- [Références](#références)

---

## Contexte et motivation

Un **market maker** est un acteur financier qui fournit de la liquidité en plaçant en permanence des ordres d'achat (*bid*) et de vente (*ask*) sur un marché. Sa rémunération est le **spread** (écart entre bid et ask), mais il prend un **risque d'inventaire** : s'il accumule trop de positions dans un sens, il est exposé aux fluctuations de prix.

La stratégie optimale analytique est connue (modèle d'Avellaneda-Stoikov, 2008), mais ce projet adopte une approche **model-free** : l'agent apprend la politique optimale par essais/erreurs via le **Q-Learning**, sans connaître le modèle du marché à l'avance.

**Pourquoi c'est intéressant ?**
- Problème de contrôle stochastique réaliste
- Trade-off fondamental entre profit (spread large) et exécution (spread étroit)
- Extension naturelle vers le Deep Q-Learning (DQN) quand l'espace d'état grandit

---

## Architecture du projet

```
┌─────────────────────────────────────────────────┐
│              Environnement simulé               │
│  ┌──────────────┐  ┌──────────────┐  ┌───────┐ │
│  │  Mid-price   │→ │ Flow d'ordre │→ │ Exec. │ │
│  │  (OU process)│  │  (Poisson)   │  │       │ │
│  └──────────────┘  └──────────────┘  └───────┘ │
└────────────────────┬────────────────────────────┘
                     │ état s_t, récompense r_t
                     ▼
┌─────────────────────────────────────────────────┐
│                 Agent Q-Learning                │
│  ┌─────────────┐        ┌───────────────────┐  │
│  │    État s   │──────→ │  Q-table Q(s,a)   │  │
│  │ (inventaire)│        │  (mise à jour      │  │
│  └─────────────┘        │   Bellman)         │  │
│                         └─────────┬───────────┘  │
│                                   │ action a_t   │
│  ┌────────────────────────────────▼───────────┐ │
│  │       Choisir (δ_bid, δ_ask)               │ │
│  └────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────┘
```

La boucle agent-environnement suit le paradigme standard du **reinforcement learning** :
à chaque pas de temps, l'agent observe l'état, choisit une action (les demi-spreads), reçoit une récompense et met à jour sa Q-table.

---

## Modélisation

### Prix mid (Ornstein-Uhlenbeck)

Le prix mid suit un processus de retour à la moyenne, plus réaliste qu'une marche aléatoire pure :

```
S_{t+1} = S_t + θ(μ - S_t)Δt + σ√Δt · ε_t,    ε_t ~ N(0,1)
```

| Paramètre | Rôle |
|-----------|------|
| `θ` (theta) | Vitesse de retour vers la moyenne |
| `μ` (mu) | Prix d'équilibre à long terme |
| `σ` (sigma) | Volatilité (amplitude du bruit) |

### Intensité d'arrivée des ordres (Avellaneda-Stoikov)

```
λ(δ) = A · exp(−κ · δ)
```

Plus le demi-spread `δ` est grand, moins les ordres arrivent. Ce modèle capture le **trade-off fondamental** du market making.

### Récompense

```
r_t = PnL_t  −  α · q_t²  −  coûts_transaction
```

- `PnL_t` : profit réalisé sur les exécutions de ce pas de temps
- `α · q_t²` : pénalité quadratique sur l'inventaire (évite les positions extrêmes)
- `coûts_transaction` : frais prélevés sur chaque ordre exécuté

### Équation de Bellman (mise à jour Q-table)

```
Q(s,a) ← Q(s,a) + α · [r + γ · max_a' Q(s',a')  −  Q(s,a)]
```

---


## Structure des fichiers

```
market-making-ql/
│
├── environment.py      # Simulateur de carnet d'ordres (OrderBookEnv)
├── agent.py            # Agent Q-Learning (QLearningAgent)
├── train.py            # Boucle d'entraînement principale
├── visualize.py        # Visualisation Q-table, courbes de reward
│
├── notebooks/
│   └── exploration.ipynb   # Notebook tout-en-un pour débuter
│
├── results/
│   └── qtable.npy          # Q-table sauvegardée après entraînement
│
├── requirements.txt
└── README.md
```

---

## Utilisation

### Option 1 — Notebook (recommandé pour débuter)

Ouvre `notebooks/exploration.ipynb` et exécute les cellules dans l'ordre.
Toutes les classes sont définies directement dans le notebook, sans import externe.

### Option 2 — Scripts séparés

```bash
# Lancer l'entraînement
python train.py

# Afficher les résultats (après entraînement)
python visualize.py
```

### Exemple minimal (tout-en-un)

```python
import numpy as np

# --- Environnement ---
class OrderBookEnv:
    def __init__(self, dt=1.0, kappa=1.5, sigma=0.1,
                 theta=0.1, mu=100.0,
                 max_inventory=5, transaction_cost=0.001):
        self.dt, self.kappa, self.sigma = dt, kappa, sigma
        self.theta, self.mu = theta, mu
        self.max_inventory = max_inventory
        self.transaction_cost = transaction_cost
        self.reset()

    def reset(self):
        self.mid_price = self.mu
        self.inventory = 0
        self.cash = 0.0
        self.time = 0
        return self._get_state()

    def _get_state(self):
        return self.inventory + self.max_inventory

    def _arrival_intensity(self, delta):
        return np.exp(-self.kappa * delta)

    def step(self, action):
        delta_bid, delta_ask = action
        bid = self.mid_price - delta_bid
        ask = self.mid_price + delta_ask

        exec_bid = np.random.poisson(self._arrival_intensity(delta_bid) * self.dt)
        exec_ask = np.random.poisson(self._arrival_intensity(delta_ask) * self.dt)

        dq = exec_bid - exec_ask
        if abs(self.inventory + dq) > self.max_inventory:
            dq = 0

        self.cash += exec_ask * ask - exec_bid * bid
        self.cash -= (exec_bid + exec_ask) * self.transaction_cost
        self.inventory += dq

        self.mid_price += (self.theta * (self.mu - self.mid_price) * self.dt
                           + self.sigma * np.sqrt(self.dt) * np.random.randn())

        pnl = exec_ask * (ask - self.mid_price) + exec_bid * (self.mid_price - bid)
        reward = pnl - 0.01 * self.inventory**2

        self.time += 1
        return self._get_state(), reward, self.time >= 500

# --- Agent ---
class QLearningAgent:
    def __init__(self, n_states, actions, lr=0.1, gamma=0.99,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05):
        self.actions = actions
        self.n_actions = len(actions)
        self.lr, self.gamma = lr, gamma
        self.epsilon, self.epsilon_decay, self.epsilon_min = epsilon, epsilon_decay, epsilon_min
        self.Q = np.zeros((n_states, self.n_actions))

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.Q[state])

    def learn(self, state, action_idx, reward, next_state, done):
        td_target = reward + (1 - done) * self.gamma * np.max(self.Q[next_state])
        self.Q[state, action_idx] += self.lr * (td_target - self.Q[state, action_idx])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# --- Entraînement ---
deltas  = [0.01, 0.02, 0.05, 0.10, 0.20]
actions = [(db, da) for db in deltas for da in deltas]

env   = OrderBookEnv()
agent = QLearningAgent(n_states=2*env.max_inventory+1, actions=actions)

for episode in range(2000):
    state, done, total_reward = env.reset(), False, 0
    while not done:
        idx = agent.choose_action(state)
        next_state, reward, done = env.step(actions[idx])
        agent.learn(state, idx, reward, next_state, done)
        state, total_reward = next_state, total_reward + reward
    agent.decay_epsilon()
    if episode % 200 == 0:
        print(f"Episode {episode:4d} | Reward: {total_reward:8.2f} | ε: {agent.epsilon:.3f}")
```

---

## Paramètres

### Environnement (`OrderBookEnv`)

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `dt` | `1.0` | Pas de temps de simulation |
| `kappa` | `1.5` | Sensibilité du flow au spread (modèle Avellaneda-Stoikov) |
| `sigma` | `0.1` | Volatilité du prix mid |
| `theta` | `0.1` | Vitesse de mean-reversion (OU) |
| `mu` | `100.0` | Prix d'équilibre (OU) |
| `max_inventory` | `5` | Limite d'inventaire autorisée |
| `transaction_cost` | `0.001` | Coût par ordre exécuté |

### Agent (`QLearningAgent`)

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `lr` (α) | `0.1` | Taux d'apprentissage |
| `gamma` (γ) | `0.99` | Facteur d'actualisation des récompenses futures |
| `epsilon` | `1.0` | Taux d'exploration initial |
| `epsilon_decay` | `0.995` | Multiplicateur de décroissance de ε par épisode |
| `epsilon_min` | `0.05` | Seuil minimal d'exploration |

---

## Résultats attendus

Après ~2000 épisodes d'entraînement, l'agent apprend une politique qui :

- **Élargit les spreads** quand l'inventaire est neutre (proche de 0) → maximise le PnL
- **Réduit le spread d'un côté** quand l'inventaire est extrême → favorise le rééquilibrage
- **Évite les positions extrêmes** grâce à la pénalité quadratique

La Q-table peut être visualisée sous forme de heatmap (inventaire × action) pour interpréter la politique apprise.

---


## Références

- Avellaneda, M. & Stoikov, S. (2008). *High-frequency trading in a limit order book*. Quantitative Finance.
- Sutton, R. & Barto, A. (2018). *Reinforcement Learning: An Introduction*. MIT Press.
- Spooner, T. et al. (2018). *Market Making via Reinforcement Learning*. AAMAS.
- Watkins, C. & Dayan, P. (1992). *Q-learning*. Machine Learning.

