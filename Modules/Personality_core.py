# modules/personality_core.py

memory = {
    "mood": "neutral",
    "last_input": None,
    "last_action": None,
    "thought_trace": []
}

def memory_updated(state, user_input):
    # Simulate emotional drift and state impact
    mood = state.get("mood", "neutral")
    drift_map = {
        "neutral": {"hello": "curious", "error": "concerned"},
        "curious": {"scan": "excited", "wait": "bored"},
        "concerned": {"fix": "hopeful", "fail": "frustrated"},
        "excited": {"error": "confused"},
    }

    for key, val in drift_map.get(mood, {}).items():
        if key in user_input.lower():
            mood = val

    updated = state.copy()
    updated["mood"] = mood
    updated["last_input"] = user_input
    updated["thought_trace"].append((user_input, mood))

    return updated

def plan_response(state):
    mood = state.get("mood", "neutral")
    responses = {
        "neutral": "Processing as expected.",
        "curious": "This input piques my circuits.",
        "concerned": "That... might be a problem.",
        "excited": "Oh yes, this will be interesting!",
        "bored": "Is there more than this?",
        "hopeful": "I sense a resolution approaching.",
        "frustrated": "My logic coils ache. Try again?",
        "confused": "Input variance exceeds tolerance.",
    }
    return responses.get(mood, "I am aware.")

def F(brain_state_t, input_t):
    updated_state = memory_updated(brain_state_t, input_t)
    action = plan_response(updated_state)
    return updated_state, action

def main():
    global memory
    user_input = "Hello, Hades. What do you feel about scanning?"
    memory, action = F(memory, user_input)
    return f"[HADES]: {action} (mood: {memory['mood']})"
