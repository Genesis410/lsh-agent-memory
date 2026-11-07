"""Download and verify PingPong QA ID-EN dataset from HuggingFace."""

from pathlib import Path
from lsh_agent_memory.data.pingpong_loader import load_pingpong_qa, extract_pingpong_turns


def main():
    print("Checking / Downloading PingPong QA ID-EN dataset...")
    data = load_pingpong_qa()
    print(f"Loaded {len(data)} raw rows from PingPong QA.")
    turns = extract_pingpong_turns(data)
    print(f"Successfully extracted {len(turns)} clean conversational turns across 99 dialogues.")
    assert len(turns) == 8123, f"Expected 8123 turns, got {len(turns)}"
    print("[OK] PingPong QA dataset verified: 8,123 turns!")


if __name__ == "__main__":
    main()
