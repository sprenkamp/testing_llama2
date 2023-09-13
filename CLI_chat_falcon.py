from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

def chat():
    model = "tiiuae/falcon-180b"

    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForCausalLM.from_pretrained(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("Chat with Girafatron! Type 'quit' to exit.")
    user_input = "Girafatron is obsessed with giraffes, the most glorious animal on the face of this Earth. Girafatron believes all other animals are irrelevant when compared to the glorious majesty of the giraffe."

    while True:
        # Get user input
        new_input = input("You: ")
        if new_input.lower() == 'quit':
            break

        user_input += f"\nYou: {new_input}\nGirafatron:"

        # Tokenize and generate response
        input_ids = tokenizer.encode(user_input, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_length=200,
                do_sample=True,
                top_k=10,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode and print Girafatron's reply
        girafatron_reply = tokenizer.decode(output[0], skip_special_tokens=True)
        last_reply = girafatron_reply.split("\n")[-1]
        print(f"Girafatron: {last_reply}")

        # Update user_input for the next round
        user_input += f" {last_reply}"

if __name__ == "__main__":
    chat()
