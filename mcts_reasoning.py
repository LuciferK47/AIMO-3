"""
Monte Carlo Tree Search (MCTS) for Mathematical Reasoning.

Implements structured search over reasoning trajectories using UCT algorithm,
PRM-guided expansion, and constrained action sampling.
"""

import logging
import math
import random
from typing import List, Dict, Optional, Tuple, Callable
import numpy as np
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class MCTSNode:
    """Node in the reasoning search tree."""
    
    state: str  # Current reasoning state (accumulated text)
    parent: Optional['MCTSNode'] = None
    children: List['MCTSNode'] = field(default_factory=list)
    visits: int = 0
    value: float = 0.0  # Total reward accumulated
    prm_score: Optional[float] = None  # PRM score for this step
    action: Optional[str] = None  # The reasoning step that led here
    is_terminal: bool = False
    final_answer: Optional[int] = None
    
    @property
    def q_value(self) -> float:
        """Average value (exploitation term)."""
        return self.value / self.visits if self.visits > 0 else 0.0
    
    def uct_score(self, exploration_constant: float = 1.41) -> float:
        """
        UCT (Upper Confidence Bound for Trees) score.
        
        Balances exploitation (average value) and exploration (visit count).
        
        Args:
            exploration_constant: Controls exploration vs exploitation (√2 typical)
            
        Returns:
            UCT score for selection
        """
        if self.visits == 0:
            return float('inf')  # Unvisited nodes prioritized
        
        if self.parent is None:
            return self.q_value
        
        exploit = self.q_value
        explore = exploration_constant * math.sqrt(math.log(self.parent.visits) / self.visits)
        
        return exploit + explore


class ReasoningMCTS:
    """
    Monte Carlo Tree Search for mathematical reasoning.
    
    Algorithm:
    1. Selection: Use UCT to traverse tree to most promising leaf
    2. Expansion: Generate candidate next reasoning steps
    3. Simulation: Complete trajectory to final answer
    4. Backpropagation: Update values using PRM and outcome
    """
    
    def __init__(self, 
                 model,
                 prm=None,
                 max_depth: int = 10,
                 exploration_const: float = 1.41,
                 num_expansions: int = 3):
        """
        Initialize MCTS.
        
        Args:
            model: Language model for generation
            prm: Process reward model for step scoring
            max_depth: Maximum reasoning depth
            exploration_const: UCT exploration parameter
            num_expansions: Number of children to expand per node
        """
        self.model = model
        self.prm = prm
        self.max_depth = max_depth
        self.exploration_const = exploration_const
        self.num_expansions = num_expansions
        
    def search(self, 
              problem: str, 
              num_simulations: int = 50,
              ground_truth: Optional[int] = None) -> Tuple[str, int]:
        """
        Run MCTS to find best reasoning trajectory.
        
        Args:
            problem: Problem text
            num_simulations: Number of MCTS iterations
            ground_truth: Correct answer (if available, for training)
            
        Returns:
            (best_trajectory, final_answer)
        """
        # Initialize root node
        root = MCTSNode(
            state=f"Problem: {problem}\n\nSolution:\n",
            parent=None
        )
        
        logger.info(f"Starting MCTS with {num_simulations} simulations")
        
        for sim in range(num_simulations):
            logger.debug(f"Simulation {sim+1}/{num_simulations}")
            
            # 1. Selection
            node = self._select(root)
            
            # 2. Expansion (if not terminal)
            if not self._is_terminal(node):
                node = self._expand(node, problem)
            
            # 3. Simulation
            reward = self._simulate(node, problem, ground_truth)
            
            # 4. Backpropagation
            self._backpropagate(node, reward)
        
        # Extract best trajectory
        best_child = self._best_child(root, exploration=0.0)
        trajectory = self._extract_trajectory(best_child)
        final_answer = best_child.final_answer if best_child else None
        
        logger.info(f"MCTS complete. Best trajectory has {len(best_child.state.split())} tokens, "
                   f"visits={best_child.visits}, value={best_child.q_value:.3f}")
        
        return trajectory, final_answer
    
    def _select(self, root: MCTSNode) -> MCTSNode:
        """
        Selection phase: traverse tree using UCT.
        
        Args:
            root: Root node to start from
            
        Returns:
            Selected leaf node for expansion
        """
        node = root
        
        while node.children and not self._is_terminal(node):
            node = self._best_child(node, exploration=self.exploration_const)
        
        return node
    
    def _best_child(self, node: MCTSNode, exploration: float) -> MCTSNode:
        """Select best child using UCT or pure exploitation."""
        if not node.children:
            return node
        
        if exploration > 0:
            # UCT selection
            scores = [child.uct_score(exploration) for child in node.children]
        else:
            # Pure exploitation (final selection)
            scores = [child.q_value for child in node.children]
        
        best_idx = np.argmax(scores)
        return node.children[best_idx]
    
    def _expand(self, node: MCTSNode, problem: str) -> MCTSNode:
        """
        Expansion phase: generate next reasoning steps.
        
        Args:
            node: Node to expand
            problem: Original problem text
            
        Returns:
            One of the newly created children
        """
        # Generate candidate next steps
        candidates = self._generate_next_steps(node, problem)
        
        # Score with PRM and keep top K
        if self.prm and candidates:
            scored = []
            for action_text in candidates:
                score = self.prm.score_reasoning_step(
                    action_text, 
                    context={'state': node.state},
                    problem_text=problem
                )
                scored.append((action_text, score))
            
            # Sort by PRM score and take top num_expansions
            scored.sort(key=lambda x: x[1], reverse=True)
            top_candidates = scored[:self.num_expansions]
        else:
            top_candidates = [(c, 0.5) for c in candidates[:self.num_expansions]]
        
        # Create child nodes
        for action_text, prm_score in top_candidates:
            child_state = node.state + action_text + "\n"
            child = MCTSNode(
                state=child_state,
                parent=node,
                action=action_text,
                prm_score=prm_score
            )
            node.children.append(child)
        
        # Return random child for simulation
        return random.choice(node.children) if node.children else node
    
    def _generate_next_steps(self, node: MCTSNode, problem: str) -> List[str]:
        """
        Generate candidate next reasoning steps.
        
        Uses model to generate K diverse continuations.
        Constrains to valid mathematical operations.
        
        Args:
            node: Current node
            problem: Problem text
            
        Returns:
            List of candidate next steps
        """
        # TODO: Implement actual model generation
        # For now, return placeholder steps
        
        # Extract current depth
        depth = len(node.state.split('\n')) - 3  # Subtract problem header
        
        # Generate template-based steps
        templates = [
            "Let's define variables for the key quantities.",
            "We can set up an equation based on the constraints.",
            "Simplifying the expression yields:",
            "Substituting the values gives:",
            "Therefore, we can conclude that:",
            "By the given condition, we have:",
            "Solving this equation step by step:",
        ]
        
        # Sample random templates
        candidates = random.sample(templates, min(5, len(templates)))
        return candidates
    
    def _simulate(self, 
                 node: MCTSNode, 
                 problem: str,
                 ground_truth: Optional[int] = None) -> float:
        """
        Simulation phase: rollout to completion.
        
        Complete trajectory from node and evaluate final answer.
        
        Args:
            node: Node to simulate from
            problem: Problem text
            ground_truth: Correct answer (if available)
            
        Returns:
            Reward [0,1]
        """
        # Check if already terminal
        if self._is_terminal(node):
            if node.final_answer is not None and ground_truth is not None:
                return 1.0 if node.final_answer == ground_truth else 0.0
            return node.prm_score if node.prm_score else 0.5
        
        # Generate completion (simplified)
        # TODO: Implement actual model completion
        
        # For now, use PRM score as proxy for quality
        if self.prm and node.action:
            return node.prm_score if node.prm_score else 0.5
        
        return 0.5  # Neutral reward
    
    def _backpropagate(self, node: MCTSNode, reward: float):
        """
        Backpropagation phase: update statistics.
        
        Args:
            node: Leaf node to backpropagate from
            reward: Reward from simulation
        """
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent
    
    def _is_terminal(self, node: MCTSNode) -> bool:
        """
        Check if node is terminal.
        
        Terminal conditions:
        - Max depth reached
        - Final answer extracted
        - Explicit termination marker
        
        Args:
            node: Node to check
            
        Returns:
            True if terminal
        """
        if node.is_terminal:
            return True
        
        # Check depth
        depth = len(node.state.split('\n')) - 3
        if depth >= self.max_depth:
            return True
        
        # Check for answer markers
        if any(marker in node.state.lower() for marker in 
               ['final answer', 'therefore the answer is', 'boxed{']):
            return True
        
        return False
    
    def _extract_trajectory(self, node: MCTSNode) -> str:
        """
        Extract full trajectory from root to node.
        
        Args:
            node: End node
            
        Returns:
            Full trajectory text
        """
        return node.state
    
    def get_tree_statistics(self, root: MCTSNode) -> Dict:
        """
        Collect statistics about the search tree.
        
        Args:
            root: Root node
            
        Returns:
            Dictionary of statistics
        """
        def traverse(node, depth=0):
            stats = {
                'total_nodes': 1,
                'max_depth': depth,
                'total_visits': node.visits,
                'terminal_nodes': 1 if self._is_terminal(node) else 0
            }
            
            for child in node.children:
                child_stats = traverse(child, depth + 1)
                stats['total_nodes'] += child_stats['total_nodes']
                stats['max_depth'] = max(stats['max_depth'], child_stats['max_depth'])
                stats['total_visits'] += child_stats['total_visits']
                stats['terminal_nodes'] += child_stats['terminal_nodes']
            
            return stats
        
        return traverse(root)


class ConstrainedActionSampler:
    """
    Sample reasoning actions constrained to valid mathematical operations.
    
    Prevents hallucinations by restricting to:
    - Algebraic manipulations
    - Substitutions
    - Known theorems
    - Arithmetic operations
    """
    
    def __init__(self):
        self.action_templates = self._init_templates()
    
    def _init_templates(self) -> Dict[str, List[str]]:
        """Initialize action templates by category."""
        return {
            'algebra': [
                "Expand the expression: {expr}",
                "Factor: {expr}",
                "Simplify: {expr}",
                "Combine like terms in {expr}",
            ],
            'substitution': [
                "Substitute {var} = {value}",
                "Let {var} = {expr}",
                "Replace {old} with {new}",
            ],
            'equation': [
                "Add {value} to both sides",
                "Multiply both sides by {value}",
                "Divide both sides by {value}",
                "Take the square root of both sides",
            ],
            'logic': [
                "By the given condition,",
                "Since {condition}, we have",
                "Therefore,",
                "It follows that",
            ]
        }
    
    def sample_action(self, context: Dict, category: Optional[str] = None) -> str:
        """
        Sample a constrained reasoning action.
        
        Args:
            context: Current reasoning context
            category: Specific category to sample from
            
        Returns:
            Reasoning action text
        """
        if category and category in self.action_templates:
            templates = self.action_templates[category]
        else:
            # Sample from all categories
            all_templates = []
            for temps in self.action_templates.values():
                all_templates.extend(temps)
            templates = all_templates
        
        # Sample random template
        template = random.choice(templates)
        
        # TODO: Fill in template with context
        return template
