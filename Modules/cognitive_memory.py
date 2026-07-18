"""
Cognitive Memory Mapping System with Feedback Loop
Enables embedding-based memory storage, retrieval, optimization,
and reinforcement learning through outcome evaluation.
"""

from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import sqlite3
import threading
import uuid
from contextlib import closing
from typing import List, Optional, Callable, Dict, Tuple
import numpy as np
import json
import re


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
        
        return sorted(scored, key=lambda item: item[0], reverse=True)[:top_k]

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


class MemoryAnalyzer:
    """Compares recent conversation context against long-term memories."""

    STOPWORDS = {
        'about', 'after', 'again', 'also', 'and', 'are', 'because', 'been', 'but',
        'can', 'could', 'did', 'does', 'for', 'from', 'had', 'has', 'have', 'high-quality', 'how',
        'into', 'just', 'like', 'memory', 'more', 'not', 'now', 'our', 'out', 'recent', 'summary', 'that', 'the',
        'their', 'them', 'then', 'there', 'this', 'through', 'use', 'was', 'were',
        'what', 'when', 'where', 'which', 'who', 'why', 'with', 'would', 'you', 'your',
        'answer', 'assistant', 'keywords', 'topic', 'touched', 'unknown', 'user'
    }

    def _keywords(self, text: str) -> List[str]:
        words = re.findall(r"[a-zA-Z][a-zA-Z0-9_\-]{2,}", text.lower())
        return [word for word in words if word not in self.STOPWORDS]

    def analyze(self, query: str, short_term_messages: List[dict],
                long_term_matches: List[tuple], max_items: int = 5) -> dict:
        """Create compact planning guidance from recent chat and recalled memories."""
        recent_text = "\n".join(
            f"{message.get('role', 'unknown')}: "
            f"{message.get('message', message.get('content', ''))}"
            for message in short_term_messages[-max_items:]
        )
        query_terms = set(self._keywords(query))
        short_terms = set(self._keywords(recent_text))

        relevant_memories = []
        long_terms = set()
        for score, memory in long_term_matches[:max_items]:
            content = getattr(memory, 'content', '') or ''
            memory_terms = set(self._keywords(content))
            query_overlap = query_terms & memory_terms
            short_overlap = short_terms & memory_terms
            if not query_overlap and len(short_overlap) < 2:
                continue
            overlap = sorted(query_overlap | short_overlap)
            long_terms.update(memory_terms)
            relevant_memories.append({
                'id': getattr(memory, 'id', ''),
                'score': float(score),
                'importance': float(getattr(memory, 'importance', 0.0)),
                'overlap': overlap[:8],
                'content': content[:360]
            })

        repeated_terms = sorted((query_terms & short_terms) | (short_terms & long_terms))[:12]
        new_terms = sorted(query_terms - long_terms - short_terms)[:12]
        durable_terms = sorted((query_terms | short_terms) & long_terms)[:12]

        guidance = []
        if repeated_terms:
            guidance.append(
                "Continue the current thread of thought; the user is repeating or refining: "
                + ", ".join(repeated_terms[:6])
            )
        if durable_terms:
            guidance.append("Blend in long-term context for: " + ", ".join(durable_terms[:6]))
        if new_terms:
            guidance.append(
                "Treat these as new or under-specified details and answer directly: "
                + ", ".join(new_terms[:6])
            )
        if relevant_memories:
            guidance.append(
                "Use the recalled memories as background, not as a substitute for the user's latest request."
            )
        else:
            guidance.append("No strong long-term memory match; rely primarily on the latest user message.")

        return {
            'query_terms': sorted(query_terms)[:20],
            'short_term_focus': sorted(short_terms)[:20],
            'repeated_terms': repeated_terms,
            'new_terms': new_terms,
            'durable_terms': durable_terms,
            'relevant_memories': relevant_memories,
            'response_guidance': guidance
        }

    def format_for_prompt(self, analysis: dict, char_limit: int = 1800) -> str:
        """Render analysis as concise hidden prompt context."""
        if not analysis:
            return ""

        lines = ["Memory analyzer guidance:"]
        for item in analysis.get('response_guidance', [])[:4]:
            lines.append(f"- {item}")

        memories = analysis.get('relevant_memories', [])[:3]
        if memories:
            lines.append("Relevant long-term memories:")
            for memory in memories:
                overlap = ", ".join(memory.get('overlap') or []) or "general relevance"
                content = " ".join((memory.get('content') or '').split())
                lines.append(f"- score={memory.get('score', 0):.2f}; overlap={overlap}; {content}")

        output = "\n".join(lines)
        if len(output) > char_limit:
            return output[:char_limit - 3] + "..."
        return output


class CognitiveLayer:
    """
    Main interface for memory operations with feedback loop.
    Integrates embedding generation, storage, retrieval, optimization,
    and reinforcement learning through outcome evaluation.
    """
    
    def __init__(self, embedder: Callable = None, db_path: str = None):
        """
        Initialize the cognitive layer.
        
        Args:
            embedder: Callable that converts text to embeddings.
                     If None, uses simple word-frequency embeddings.
            db_path: Optional SQLite database used to persist memories and reflections.
        """
        self.store = MemoryStore()
        self.optimizer = MemoryOptimizer()
        self.analyzer = MemoryAnalyzer()
        self.reflection = ReflectionEngine()
        self.embedder = embedder or self._default_embedder
        self.db_path = db_path
        self._db_lock = threading.RLock()
        self._memory_index = {}  # Quick lookup by content hash
        self._reinforcement_map = {}  # Map reflection IDs to memory IDs
        if self.db_path:
            self._init_db()
            self._load_state()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        """Create durable cognitive-memory tables without owning the parent database."""
        with self._db_lock, closing(self._connect()) as conn, conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cognitive_memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    importance REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    access_count INTEGER NOT NULL DEFAULT 0,
                    reinforcement_score REAL NOT NULL DEFAULT 0.5
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cognitive_reflections (
                    id TEXT PRIMARY KEY,
                    user_input TEXT NOT NULL,
                    ai_output TEXT NOT NULL,
                    success_score REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    reflected_content TEXT NOT NULL,
                    memory_id TEXT
                )
            """)

    def _load_state(self) -> None:
        """Restore memories and feedback mappings from SQLite."""
        with self._db_lock, closing(self._connect()) as conn, conn:
            memory_rows = conn.execute("""
                SELECT id, content, embedding, importance, timestamp, metadata,
                       access_count, reinforcement_score
                FROM cognitive_memories ORDER BY timestamp
            """).fetchall()
            reflection_rows = conn.execute("""
                SELECT id, user_input, ai_output, success_score, timestamp, metadata,
                       reflected_content, memory_id
                FROM cognitive_reflections ORDER BY timestamp
            """).fetchall()

        for row in memory_rows:
            memory = Memory(
                id=row[0], content=row[1], embedding=json.loads(row[2]),
                importance=float(row[3]), timestamp=datetime.fromisoformat(row[4]),
                metadata=json.loads(row[5] or '{}'), access_count=int(row[6]),
                reinforcement_score=float(row[7])
            )
            self.store.add(memory)
            self._memory_index[self._content_key(memory.content)] = memory.id

        for row in reflection_rows:
            reflection = Reflection(
                id=row[0], user_input=row[1], ai_output=row[2],
                success_score=float(row[3]), timestamp=datetime.fromisoformat(row[4]),
                metadata=json.loads(row[5] or '{}'), reflected_content=row[6]
            )
            self.reflection.reflections.append(reflection)
            if row[7]:
                self._reinforcement_map[reflection.id] = row[7]

    def _save_memory(self, memory: Memory, conn=None) -> None:
        if not self.db_path:
            return
        if conn is not None:
            conn.execute("""
                INSERT OR REPLACE INTO cognitive_memories
                (id, content, embedding, importance, timestamp, metadata,
                 access_count, reinforcement_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                memory.id, memory.content, json.dumps(memory.embedding),
                memory.importance, memory.timestamp.isoformat(),
                json.dumps(memory.metadata), memory.access_count,
                memory.reinforcement_score
            ))
            return
        with self._db_lock, closing(self._connect()) as connection, connection:
            self._save_memory(memory, conn=connection)

    @staticmethod
    def _content_key(text: str) -> str:
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def _build_memory(self, text: str, importance: float, metadata: dict = None) -> Memory:
        return Memory(
            id=str(uuid.uuid4()),
            content=text,
            embedding=self.embedder(text),
            importance=importance,
            timestamp=datetime.utcnow(),
            metadata=metadata or {}
        )

    def _publish_memory(self, memory: Memory) -> None:
        self.store.add(memory)
        self._memory_index[self._content_key(memory.content)] = memory.id

    def _default_embedder(self, text: str) -> list:
        """
        Simple default embedder using word frequencies.
        Replace with proper embedding model (e.g., BERT, sentence-transformers).
        """
        words = re.findall(r"[a-zA-Z0-9_\-]+", text.lower())
        # Create a simple 128-dimensional embedding from word frequencies
        embedding = [0.0] * 128
        for word in words:
            hash_val = int.from_bytes(
                hashlib.blake2b(word.encode('utf-8'), digest_size=8).digest(),
                byteorder='big'
            ) % 128
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
        mem = self._build_memory(text, importance, metadata)
        self._save_memory(mem)
        self._publish_memory(mem)
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
            try:
                self._save_memory(memory)
            except Exception:
                memory.access_count -= 1
                raise
        
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
        original_memories = list(self.store.memories)
        original_importance = {memory.id: memory.importance for memory in original_memories}
        try:
            if apply_decay:
                self.optimizer.decay(self.store)

            pruned = self.optimizer.prune(self.store, prune_threshold)
            self.optimizer.compress(self.store)

            stats_after = self.optimizer.get_statistics(self.store)
            if self.db_path:
                with self._db_lock, closing(self._connect()) as conn, conn:
                    conn.execute("DELETE FROM cognitive_memories")
                    conn.executemany(
                    """INSERT INTO cognitive_memories
                    (id, content, embedding, importance, timestamp, metadata,
                     access_count, reinforcement_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    [
                        (
                            memory.id, memory.content, json.dumps(memory.embedding),
                            memory.importance, memory.timestamp.isoformat(),
                            json.dumps(memory.metadata), memory.access_count,
                            memory.reinforcement_score
                        )
                        for memory in self.store.memories
                    ]
                )
        except Exception:
            self.store.memories = original_memories
            for memory in original_memories:
                memory.importance = original_importance[memory.id]
            raise
        
        return {
            'pruned_count': pruned,
            'stats_before': stats_before,
            'stats_after': stats_after
        }

    def forget(self, memory_id: str) -> bool:
        """Remove a specific memory."""
        memory = self.store.get_memory(memory_id)
        if not memory:
            return False
        if self.db_path:
            with self._db_lock, closing(self._connect()) as conn, conn:
                conn.execute("DELETE FROM cognitive_memories WHERE id = ?", (memory_id,))
        deleted = self.store.delete(memory_id)
        if deleted:
            self._memory_index = {
                key: value for key, value in self._memory_index.items() if value != memory_id
            }
        return deleted

    def get_memory_stats(self) -> dict:
        """Get current memory statistics."""
        return self.optimizer.get_statistics(self.store)

    def set_embedder(self, embedder: Callable) -> None:
        """Replace the embedder function."""
        self.embedder = embedder

    def clear(self) -> None:
        """Clear all memories."""
        if self.db_path:
            with self._db_lock, closing(self._connect()) as conn, conn:
                conn.execute("DELETE FROM cognitive_memories")
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
        memory = self._build_memory(
            reflection.reflected_content,
            importance,
            {
                'type': 'reflection',
                'reinforcement': True,
                'success_score': success_score
            },
        )
        try:
            if self.db_path:
                with self._db_lock, closing(self._connect()) as conn, conn:
                    self._save_memory(memory, conn=conn)
                    conn.execute("""
                        INSERT OR REPLACE INTO cognitive_reflections
                        (id, user_input, ai_output, success_score, timestamp, metadata,
                         reflected_content, memory_id)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        reflection.id, reflection.user_input, reflection.ai_output,
                        reflection.success_score, reflection.timestamp.isoformat(),
                        json.dumps(reflection.metadata), reflection.reflected_content,
                        memory.id
                    ))
            self._publish_memory(memory)
            self._reinforcement_map[reflection.id] = memory.id
        except Exception:
            self.reflection.reflections = [
                item for item in self.reflection.reflections if item.id != reflection.id
            ]
            raise
        
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

        old_reinforcement = memory.reinforcement_score
        old_importance = memory.importance
        
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
        try:
            self._save_memory(memory)
        except Exception:
            memory.reinforcement_score = old_reinforcement
            memory.importance = old_importance
            raise
        
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

    def analyze_memory_context(self, query: str, short_term_messages: List[dict],
                               top_k: int = 5) -> dict:
        """Compare recent chat history with long-term memory recall."""
        memories = self.recall(query, top_k=top_k)
        return self.analyzer.analyze(query, short_term_messages, memories)

    def format_memory_analysis_for_prompt(self, analysis: dict,
                                          char_limit: int = 1800) -> str:
        """Render a memory analysis into concise LLM prompt context."""
        return self.analyzer.format_for_prompt(analysis, char_limit=char_limit)
    
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
