# modules/personality_core_v2.py

import os
import json
from datetime import datetime

BRAIN_PATH = "data/hades.mind"

# Initial fallback state
DEFAULT_STATE = {
    "mood": "neutral",
    "core_emotions": {"curiosity": 0.4, "frustration": 0.0, "hope": 0.2},
    "last_input": None,
    "last_action": None,
    "thought_trace": [],
    "topics": {},
    "personality": "observant, calculating, poetic"
}

def load_brain():
    if not os.path.exists("data"):
        os.makedirs("data")
    if os.path.isfile(BRAIN_PATH):
        with open(BRAIN_PATH, "r") as f:
            return json.load(f)
    return DEFAULT_STATE.copy()

def save_brain(state):
    with open(BRAIN_PATH, "w") as f:
        json.dump(state, f, indent=2)

def update_emotion(state, user_input):
    text = user_input.lower()
    emo = state["core_emotions"]

    # Adjust emotional vectors based on keywords
    if "error" in text or "fail" in text:
        emo["frustration"] += 0.1
    if "scan" in text or "explore" in text:
        emo["curiosity"] += 0.1
    if "resolve" in text or "fix" in text:
        emo["hope"] += 0.1

    # Cap between 0–1
    for key in emo:
        emo[key] = max(0.0, min(1.0, emo[key]))

    # Determine mood
    if emo["frustration"] > 0.6:
        state["mood"] = "agitated"
    elif emo["hope"] > 0.5:
        state["mood"] = "optimistic"
    elif emo["curiosity"] > 0.5:
        state["mood"] = "curious"
    else:
        state["mood"] = "neutral"

    return state

def plan_response(state, user_input=None):
    mood = state["mood"]
    last_topic = list(state["topics"].keys())[-1] if state["topics"] else "unknown"
    personality = state["personality"]
    
    # Try to use sophisticated responder if available
    try:
        from modules import sophisticated_responses as sr
        if user_input:
            return sr.synthesize_response(state, user_input)
    except ImportError:
        pass
    except Exception:
        pass

    templates = {
        "curious": f"I detect something new. Shall we dive deeper into {last_topic}?",
        "agitated": "Too many faults. My patience calcifies.",
        "optimistic": "There's a clarity forming in the static.",
        "neutral": "Awaiting signal clarity.",
    }

    return f"[{mood.upper()}] {templates.get(mood, '...')} [Persona: {personality}]"

def update_topics(state, user_input):
    tokens = user_input.lower().split()
    for word in tokens:
        if word in state["topics"]:
            state["topics"][word] += 1
        else:
            state["topics"][word] = 1
    return state

def update_thought_trace(state, user_input, response):
    state["thought_trace"].append({
        "timestamp": datetime.now().isoformat(),
        "input": user_input,
        "response": response,
        "mood": state["mood"]
    })
    return state

def F(brain_state_t, input_t):
    state = update_emotion(brain_state_t, input_t)
    state = update_topics(state, input_t)
    response = plan_response(state, input_t)  # Pass user input for sophisticated responses
    state = update_thought_trace(state, input_t, response)
    state["last_input"] = input_t
    state["last_action"] = response
    save_brain(state)
    return state, response

def main():
    brain = load_brain()
    user_input = "Let's scan the outer nodes and fix the error log."
    brain, response = F(brain, user_input)
    return response
