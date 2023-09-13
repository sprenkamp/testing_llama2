from transformers import AutoTokenizer
import transformers
import torch

def get_recommendations(prompt):
    model = "meta-llama/Llama-2-7b-chat-hf"

    tokenizer = AutoTokenizer.from_pretrained(model)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    sequences = pipeline(
        prompt,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=200,
    )
    for seq in sequences:
        print(f"Result: {seq['generated_text']}")

if __name__ == "__main__":
    while True:
        user_input = input("Ask the bot something (or type 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break
        get_recommendations(user_input)
