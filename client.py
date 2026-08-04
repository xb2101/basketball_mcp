import sys
sys.path.append('.')
from utils import chat

def main():
    print("Basketball RL MCP Chatbot")
    print("Ask me anything about the training data!")
    print("Type 'quit' to exit\n")
    
    conversation_history = []
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() == "quit":
            break
            
        if not user_input:
            continue
        
        print("\nClaude: ", end="")
        response = chat(user_input, conversation_history)
        print(response)
        print()

if __name__ == "__main__":
    main()