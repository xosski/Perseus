"""
Cognitive Memory Mapping System with Feedback Loop
Enables embedding-based memory storage, retrieval, optimization,
and reinforcement learning through outcome evaluation.
"""

from dataclasses import dataclass, field
from datetime import datetime
import uuid
from typing import List, Optional, Callable, Dict, Tuple
import numpy as np
import json


@dataclass
class Memory:
    """Represents a single memory entry with embedding and metadata."""
    id: str
    content: str
    embedding: list
    importance: float
    timestamp: datetime
    metadata: dict = field(default_factory=dict)
    access_count: int = 0  # Track how many times recalled
    reinforcement_score: float = 0.5  # Track feedback reinforcement


@dataclass
class Reflection:
    """Represents an outcome evaluation and reflection."""
    id: str
    user_input: str
    ai_output: str
    success_score: float  # 0.0-1.0
    timestamp: datetime
    metadata: dict = field(default_factory=dict)
    reflected_content: str = ""  # Structured reflection text


class MemoryStore:
    """In-memory storage for memories with cosine similarity search."""
    
    def __init__(self):
        self.memories: List[Memory] = []

    def add(self, memory: Memory) -> None:
        """Add a memory to the store."""
        self.memories.append(memory)

    def search(self, query_embedding: list, top_k: int = 5, use_reinforcement: bool = True) -> List[tuple]:
        """
        Search memories by embedding similarity with optional reinforcement bias.
        Returns list of (similarity_score, Memory) tuples.
        """
        def cosine_similarity(a, b):
            a_norm = np.linalg.norm(a)
            b_norm = np.linalg.norm(b)
            if a_norm == 0 or b_norm == 0:
                return 0.0
            return float(np.dot(a, b) / (a_norm * b_norm))
        
        scored = []
        for m in self.memories:
            sim = cosine_similarity(query_embedding, m.embedding)
            # Apply reinforcement bias: boost memories with higher reinforcement scores
            if use_reinforcement:
                sim = sim * (0.7 + 0.3 * m.reinforcement_score)
            scored.append((sim, m))
        
        return sorted(scored, reverse=True)[:top_k]

    def get_memory(self, memory_id: str) -> Optional[Memory]:
        """Retrieve a specific memory by ID."""
        for m in self.memories:
            if m.id == memory_id:
                return m
        return None

    def delete(self, memory_id: str) -> bool:
        """Remove a memory from storage."""
        initial_len = len(self.memories)
        self.memories = [m for m in self.memories if m.id != memory_id]
        return len(self.memories) < initial_len

    def size(self) -> int:
        """Return total number of stored memories."""
        return len(self.memories)


class ReflectionEngine:
    """Handles outcome evaluation and memory reinforcement."""
    
    def __init__(self):
        self.reflections: List[Reflection] = []
    
    def create_reflection(self, user_input: str, ai_output: str, 
                         success_score: float, metadata: dict = None) -> Reflection:
        """
        Create a reflection from an interaction outcome.
        
        Args:
            user_input: Original user input
            ai_output: AI response
            success_score: Outcome evaluation (0.0-1.0)
            metadata: Optional metadata
            
        Returns:
            Reflection object
        """
        reflected_content = f"""
Input: {user_input}
Output: {ai_output}
Outcome Score: {success_score:.2f}
Timestamp: {datetime.utcnow()}
"""
        reflection = Reflection(
            id=str(uuid.uuid4()),
            user_input=user_input,
            ai_output=ai_output,
            success_score=success_score,
            timestamp=datetime.utcnow(),
            metadata=metadata or {},
            reflected_content=reflected_content.strip()
        )
        self.reflections.append(reflection)
        return reflection
    
    def calculate_reinforcement(self, success_score: float, base_importance: float = 0.3) -> float:
        """
        Calculate memory importance based on reinforcement feedback.
        
        Args:
            success_score: Outcome score (0.0-1.0)
            base_importance: Base importance level
            
        Returns:
            Adjusted importance (0.0-1.0)
        """
        return min(1.0, base_importance + success_score * 0.7)
    
    def get_reflection_stats(self) -> dict:
        """Get statistics about reflections."""
        if not self.reflections:
            return {
                'total_reflections': 0,
                'avg_success': 0.0,
                'best_success': 0.0,
                'worst_success': 0.0
            }
        
        scores = [r.success_score for r in self.reflections]
        return {
            'total_reflections': len(self.reflections),
            'avg_success': float(np.mean(scores)),
            'best_success': float(np.max(scores)),
            'worst_success': float(np.min(scores)),
            'recent': [
                {
                    'input': r.user_input[:100],
                    'score': r.success_score,
                    'timestamp': r.timestamp.isoformat()
                }
                for r in self.reflections[-5:]
            ]
        }


class MemoryOptimizer:
    """Handles memory pruning and compression."""
    
    def prune(self, store: MemoryStore, threshold: float = 0.2) -> int:
        """
        Remove low-importance memories below threshold.
        Returns count of pruned memories.
        """
        initial_count = store.size()
        store.memories = [
            m for m in store.memories
            if m.importance >= threshold
        ]
        return initial_count - store.size()

    def compress(self, store: MemoryStore) -> None:
        """
        Compress similar memories by averaging embeddings.
        Placeholder for future implementation with clustering.
        """
        # TODO: Implement clustering-based compression
        # Group similar memories, summarize content, average embeddings
        pass

    def decay(self, store: MemoryStore, decay_rate: float = 0.95) -> None:
        """
        Apply time-based decay to memory importance.
        Older memories gradually lose importance.
        """
        now = datetime.utcnow()
        for memory in store.memories:
            age_days = (now - memory.timestamp).days
            decay_factor = decay_rate ** (age_days / 7)  # Decay per week
            memory.importance *= decay_factor

    def get_statistics(self, store: MemoryStore) -> dict:
        """Get statistics about stored memories."""
        if not store.memories:
            return {
                'total_memories': 0,
                'avg_importance': 0.0,
                'oldest': None,
                'newest': None
            }
        
        importances = [m.importance for m in store.memories]
        timestamps = [m.timestamp for m in store.memories]
        
        return {
            'total_memories': store.size(),
            'avg_importance': np.mean(importances),
            'max_importance': float(np.max(importances)),
            'min_importance': float(np.min(importances)),
            'oldest': min(timestamps),
            'newest': max(timestamps)
        }


class CognitiveLayer:
    """
    Main interface for memory operations with feedback loop.
    Integrates embedding generation, storage, retrieval, optimization,
    and reinforcement learning through outcome evaluation.
    """
    
    def __init__(self, embedder: Callable = None):
        """
        Initialize the cognitive layer.
        
        Args:
            embedder: Callable that converts text to embeddings.
                     If None, uses simple word-frequency embeddings.
        """
        self.store = MemoryStore()
        self.optimizer = MemoryOptimizer()
        self.reflection = ReflectionEngine()
        self.embedder = embedder or self._default_embedder
        self._memory_index = {}  # Quick lookup by content hash
        self._reinforcement_map = {}  # Map reflection IDs to memory IDs

    def _default_embedder(self, text: str) -> list:
        """
        Simple default embedder using word frequencies.
        Replace with proper embedding model (e.g., BERT, sentence-transformers).
        """
        words = text.lower().split()
        # Create a simple 128-dimensional embedding from word frequencies
        embedding = [0.0] * 128
        for word in words:
            hash_val = hash(word) % 128
            embedding[hash_val] += 1.0 / (len(words) + 1)
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = [x / norm for x in embedding]
        return embedding

    def remember(self, text: str, importance: float = 0.5, metadata: dict = None) -> str:
        """
        Store a new memory.
        
        Args:
            text: Memory content
            importance: Importance score (0.0-1.0)
            metadata: Optional metadata dictionary
            
        Returns:
            Memory ID
        """
        embedding = self.embedder(text)
        mem = Memory(
            id=str(uuid.uuid4()),
            content=text,
            embedding=embedding,
            importance=importance,
            timestamp=datetime.utcnow(),
            metadata=metadata or {}
        )
        self.store.add(mem)
        self._memory_index[hash(text)] = mem.id
        return mem.id

    def recall(self, query: str, top_k: int = 5, use_reinforcement: bool = True) -> List[tuple]:
        """
        Retrieve similar memories with optional reinforcement bias.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            use_reinforcement: Apply reinforcement bias to boost successful memories
            
        Returns:
            List of (similarity_score, Memory) tuples
        """
        query_embedding = self.embedder(query)
        results = self.store.search(query_embedding, top_k, use_reinforcement)
        
        # Track access for memory usage patterns
        for _, memory in results:
            memory.access_count += 1
        
        return results

    def optimize(self, prune_threshold: float = 0.2, apply_decay: bool = True) -> dict:
        """
        Optimize memory storage.
        
        Args:
            prune_threshold: Remove memories below this importance
            apply_decay: Apply time-based decay
            
        Returns:
            Optimization statistics
        """
        stats_before = self.optimizer.get_statistics(self.store)
        
        if apply_decay:
            self.optimizer.decay(self.store)
        
        pruned = self.optimizer.prune(self.store, prune_threshold)
        self.optimizer.compress(self.store)
        
        stats_after = self.optimizer.get_statistics(self.store)
        
        return {
            'pruned_count': pruned,
            'stats_before': stats_before,
            'stats_after': stats_after
        }

    def forget(self, memory_id: str) -> bool:
        """Remove a specific memory."""
        return self.store.delete(memory_id)

    def get_memory_stats(self) -> dict:
        """Get current memory statistics."""
        return self.optimizer.get_statistics(self.store)

    def set_embedder(self, embedder: Callable) -> None:
        """Replace the embedder function."""
        self.embedder = embedder

    def clear(self) -> None:
        """Clear all memories."""
        self.store.memories.clear()
        self._memory_index.clear()
    
    # ========== Feedback Loop Methods ==========
    
    def evaluate_response(self, user_input: str, ai_output: str, 
                         success_score: float, metadata: dict = None) -> str:
        """
        Evaluate an AI response and create a reflection.
        This is the "outcome evaluation" step in the feedback loop.
        
        Args:
            user_input: User's original query
            ai_output: AI's response
            success_score: Evaluation score (0.0-1.0)
            metadata: Optional metadata about the interaction
            
        Returns:
            Reflection ID
        """
        reflection = self.reflection.create_reflection(
            user_input, ai_output, success_score, metadata
        )
        
        # Optionally store the reflection itself as a memory
        importance = self.reflection.calculate_reinforcement(success_score)
        memory_id = self.remember(
            text=reflection.reflected_content,
            importance=importance,
            metadata={
                'type': 'reflection',
                'reinforcement': True,
                'success_score': success_score
            }
        )
        self._reinforcement_map[reflection.id] = memory_id
        
        return reflection.id
    
    def reinforce_memory(self, memory_id: str, success_score: float) -> bool:
        """
        Reinforce a specific memory based on feedback.
        Updates the memory's reinforcement score and importance.
        
        Args:
            memory_id: Memory to reinforce
            success_score: Feedback score (0.0-1.0)
            
        Returns:
            Success status
        """
        memory = self.store.get_memory(memory_id)
        if not memory:
            return False
        
        # Update reinforcement score with exponential moving average
        alpha = 0.3  # Learning rate
        memory.reinforcement_score = (
            alpha * success_score + (1 - alpha) * memory.reinforcement_score
        )
        
        # Adjust importance based on reinforcement
        new_importance = self.reflection.calculate_reinforcement(
            memory.reinforcement_score,
            base_importance=0.3
        )
        memory.importance = max(memory.importance, new_importance)
        
        return True
    
    def generate_with_memory(self, query: str, context_provider: Callable) -> Tuple[str, List[tuple]]:
        """
        Generate a response using recalled memories as context.
        This implements memory-augmented generation.
        
        Args:
            query: User query
            context_provider: Callable that takes (query, memory_context) -> response
                             This is typically your LLM call
            
        Returns:
            Tuple of (response, recalled_memories)
        """
        # Recall relevant memories
        memories = self.recall(query, top_k=5)
        
        # Format memory context
        memory_context = "\n".join(
            f"- {m.content}" for _, m in memories
        )
        
        # Generate response with memory context
        response = context_provider(query, memory_context)
        
        return response, memories
    
    def get_reflection_stats(self) -> dict:
        """Get statistics about reflections and reinforcement."""
        return self.reflection.get_reflection_stats()
    
    def get_full_stats(self) -> dict:
        """Get comprehensive statistics about memories and reflections."""
        memory_stats = self.get_memory_stats()
        reflection_stats = self.get_reflection_stats()
        
        # Calculate memory quality metrics
        if self.store.memories:
            access_counts = [m.access_count for m in self.store.memories]
            reinforcement_scores = [m.reinforcement_score for m in self.store.memories]
            
            memory_stats['avg_access_count'] = float(np.mean(access_counts))
            memory_stats['avg_reinforcement'] = float(np.mean(reinforcement_scores))
            memory_stats['max_reinforcement'] = float(np.max(reinforcement_scores))
        
        return {
            'memories': memory_stats,
            'reflections': reflection_stats,
            'integration_quality': {
                'reinforced_memories': sum(
                    1 for m in self.store.memories 
                    if m.reinforcement_score > 0.5
                ),
                'frequently_accessed': sum(
                    1 for m in self.store.memories 
                    if m.access_count > 2
                )
            }
        }
