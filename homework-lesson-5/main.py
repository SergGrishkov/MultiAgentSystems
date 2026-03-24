from agent import agent
from tools import write_report
from datetime import datetime


def main():
    print("Research Agent (type 'exit' to quit)")
    print("-" * 40)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        # Configure with thread_id for memory
        config = {"configurable": {"thread_id": "default"}}

        responses = []
        for chunk in agent.stream(
            {"messages": [("user", user_input)]},
            config=config,
        ):
            if "agent" in chunk and "messages" in chunk["agent"]:
                for msg in chunk["agent"]["messages"]:
                    if hasattr(msg, "content") and msg.content:
                        text = msg.content.strip()
                        responses.append(text)
                        print(f"\nAgent: {text}")

        if responses:
            report_content = f"# User query\n{user_input}\n\n# Agent response\n" + "\n\n".join(responses)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"report_{timestamp}.md"
            save_msg = write_report(report_filename, report_content)
            print(f"\n{save_msg}")


if __name__ == "__main__":
    main()
