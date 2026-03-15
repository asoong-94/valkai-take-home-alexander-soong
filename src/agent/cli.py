import argparse
import uuid

from dotenv import load_dotenv

from agent.strategies import REGISTRY, make_strategy


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="CLI Chat Agent")
    parser.add_argument(
        "--model",
        default="anthropic:claude-haiku-4-5-20251001",
        help="Model string, e.g. openai:gpt-4o, anthropic:claude-haiku-4-5-20251001, google_genai:gemini-2.5-flash",
    )
    parser.add_argument(
        "--memory",
        choices=list(REGISTRY),
        default="baseline",
        help="Memory strategy (default: baseline)",
    )
    parser.add_argument(
        "--user-id",
        default="default-user",
        help="User ID for cross-session memory (default: default-user)",
    )
    parser.add_argument(
        "--data-dir",
        default="./data",
        help="Directory for persistent memory storage (default: ./data)",
    )
    args = parser.parse_args()

    strategy = make_strategy(args.memory, args.model, data_dir=args.data_dir)
    thread_id = uuid.uuid4().hex[:12]

    print(f"Strategy: {strategy.name} | Model: {args.model}")
    print(f"User: {args.user_id} | Thread: {thread_id}")
    print("Type 'quit' to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            break

        reply = strategy.chat(user_input, user_id=args.user_id, thread_id=thread_id)
        print(f"\nAssistant: {reply}\n")


if __name__ == "__main__":
    main()
