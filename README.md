# Reinforcement Learning for Marvel Snap: Agent Design, Training and Evaluation

This project applies Reinforcement Learning (RL) to build an agent that plays the digital collectible card game **Marvel Snap** (simulated), with a particular focus on generalizing to decks built from cards unseen during training. The work was developed as a Master's thesis in Computer Engineering at the University of Trieste.

## Why Marvel Snap?

Digital collectible card games have already been studied extensively as RL benchmarks (e.g. Hearthstone), because they combine a huge state space, partial observability, stochasticity, and strong dependence on card synergies. Marvel Snap adds a distinctive mechanic on top of these challenges: **stake management**. Each match has a variable number of points ("cubes") at stake, and once per game each player may choose to double the stakes ("Snap"), retreat to cut their losses, or do nothing. This turns Marvel Snap into two intertwined problems: playing well tactically, and managing risk strategically.

## Two Separate Agents

A single agent trying to both win matches and manage the stakes optimally would have to reconcile conflicting objectives inside one action space — for instance, retreating is never useful if the sole goal is winning the match, but it can be the best move if the goal is minimizing lost cubes. To avoid this conflict, the problem is split into **two independent RL problems**, each with its own state space, action space and reward function:

- **Player Agent** — decides, at each step, which card to play and where, or whether to pass the turn. Its objective is simply to win the match.
- **Better Agent** — decides, once per turn, whether to Snap, retreat, or do nothing. Its objective is to maximize cubes won and minimize cubes lost.

The Better Agent internally relies on a trained Player Agent to actually play out the cards; it only makes stake decisions on top of the resulting match. Both problems are trained separately, with the Player Agent trained first and its frozen policy later reused as the "card-playing engine" for the Better Agent's training.

## Problem Formalization

Both agents are modeled as **Partially Observable Markov Decision Processes (POMDPs)**, since neither agent has full knowledge of the game state (e.g. the opponent's hand).

**State.** The full game state includes general match information (turn number, energy available, priority), the three active locations (each with per-player strength and a textual effect description), the cards in both players' hands, and the cards currently on the board. Every card is represented by its strength, energy cost, and textual effect description. The Better Agent's state additionally includes the cubes currently at stake, the cubes that will be at stake next turn, and flags indicating whether each player has already Snapped.

**Observation.** The observation is a restricted view of the state: in this case it omits the cards in the opponent's hand and the opponent's available energy.

**Action.** For the Player Agent, an action is either "play card *i* from hand at location *j*" or "pass the turn," giving a fixed action space of 22 actions (though only a subset is valid in a given state, due to energy constraints, hand size, etc.). For the Better Agent, the action space has just three members: Snap, retreat, do nothing (with Snap excluded from the legal set once already used).

**Reward.** The Player Agent receives a *terminal reward* (positive/negative/zero depending on win/loss/draw) and a small *intermediate reward* at the end of each turn based on how many locations are currently won or lost — scaled up as the match progresses, so it nudges strategy early on without overriding the win/loss objective near the end. The Better Agent receives only a terminal reward equal to the number of cubes won or lost, since there is no meaningful way to attribute partial credit to intermediate stake decisions.

## Representing Cards for Generalization

A key design choice is how cards and locations are represented in the state, since the goal is to generalize to cards never seen in training. Instead of using card identifiers (which cannot generalize to new cards), each card's and location's textual effect description is embedded using a pretrained sentence-embedding model (MPNet, 768 features). This lets the network learn associations between *semantically similar* effects, rather than between specific IDs — at the cost of a much larger input vector. This is a deliberate efficiency-for-generalization trade-off: training is slower, but the resulting policy can, in principle, handle unseen cards and locations whose effects resemble those seen during training.

Actions are encoded directly alongside the observation as part of the network's input, rather than as a separate output layer: the Player Agent's action is represented via two binary flags (which card and which location were chosen, or -1/-1 for passing); the Better Agent's action is a one-hot vector over its three choices.

## Algorithm: Deep Monte Carlo

Given the scale and stochasticity of the state space, table-based methods are infeasible. The project uses **Deep Monte Carlo (DMC)**, previously shown effective on similarly complex card games (DouDizhu, Hearthstone). The core idea:

1. A neural network approximates the optimal action-value function Q*(s, a) — instead of a full lookup table, the network takes a (state, action) pair as input and predicts its expected discounted return.
2. Full episodes (matches) are simulated to completion, recording every (observation, action) pair encountered.
3. Once an episode ends, the *actually observed* discounted return G_t is computed for every step, and the network is trained to minimize the mean squared error between its predicted Q-value and G_t.

Both agents use a feed-forward network with four hidden layers of 512 units each (the wide input layer is kept small on purpose, since input size dominates the parameter count).

**Exploration vs. exploitation** is handled via an ε-greedy policy: with probability ε, an action is drawn uniformly at random from the legal actions; otherwise, the network's highest-scoring legal action is chosen. ε decays linearly over training, from a high initial value (favoring exploration early on) to a small final value.

### Player Agent training specifics

The opponent faced during self-play also uses an ε-greedy policy, but with its own decay schedule: it starts fully random (ε=1) and gradually converges toward the same policy used by the agent (ε=0.05), reaching its floor around the midpoint of training. This produces a mix between a random-opponent curriculum and full self-play, avoiding both an opponent that is too weak to teach anything and a fragile monoculture where the agent only ever learns to beat itself. The discount factor is set to 1 (no discounting), since intermediate rewards are sparse and rewards further from the terminal state should not be penalized relative to more immediate ones. Four handcrafted decks are used during training, randomly paired at the start of each match, to encourage the agent to generalize across different card combinations rather than overfitting to a single deck.

### Better Agent training specifics

The Better Agent faces a fixed heuristic opponent that Snaps with a probability growing linearly with the turn number (mimicking the tendency of human players to escalate stakes later in a match), and never retreats. During training, the Better Agent's own exploration does not sample uniformly among its three actions (which would cause an unrealistically high retreat rate); instead it uses a hand-crafted probability distribution biased heavily toward "do nothing," closer to what a reasonable player would actually do. The discount factor here is set to 0.9 rather than 1, reflecting the intuition that decisions made in the final turns of a match are more directly tied to the eventual stake outcome than decisions made early on.

## Evaluation Methodology

Both agents are evaluated periodically during training by pausing training, disabling exploration (ε=0), and playing a batch of evaluation matches.

- The **Player Agent** is scored by win rate against a random-move baseline opponent (50% expected win rate between two random players).
- The **Better Agent** is scored by average cubes won per match, since it always plays against a mirror of itself at a roughly 50% win rate — win rate alone would not capture whether stake decisions are actually good.

After training, both agents are additionally benchmarked against a **greedy heuristic opponent** (which plays its highest-strength card at the location where the strength gap is smallest, and Snaps/retreats only on the final turn based on whether it is winning), and tested with decks containing cards partially or fully unseen during training, to assess generalization.

## Summary of Results

- The Player Agent improves substantially over a random opponent (win rate rising from ~50% to ~76% over 2M training episodes) and retains most of this advantage even with fully unseen decks (~69% win rate). Against the greedy heuristic, however, the advantage shrinks considerably (~52–56% win rate), suggesting the current setup captures general competence but not a decisive tactical edge.
- The Better Agent increases its average cubes won substantially over training (from ~0 to ~0.7), primarily by learning *when to retreat* rather than by improving its ability to Snap. It generalizes well to partially unseen decks but essentially fails to generalize to decks built entirely from unseen cards, where average cubes won drops close to zero.
- Across both agents, deck composition has a large effect on results: some decks are consistently stronger than others, and specific deck matchups can shift win rates by double-digit percentage points.

## Limitations and Future Work

The project is constrained by the simulator's coverage of the full card and location pool, and by the multi-day computational cost of a single DMC training run, which prevented repeated runs for statistical robustness. Suggested extensions include reducing the embedding dimensionality to speed up training, parallelizing episode simulation across multiple cores while keeping network updates centralized, and enriching the state representation with deck composition and unrevealed-but-played cards.

## Thesis

The complete thesis describing the project can be found [here](https://drive.google.com/file/d/1z8HhekVkdYdv_JcUA9PsCNyWiWYK8h-0/view).

## Author

Daniele Maijnelli
