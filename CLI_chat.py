from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import os

HF_TOKEN = os.environ["HF_TOKEN"] 

def generate_with_falcon(prompt):
    model_id = "tiiuae/falcon-180B"

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=True, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        use_auth_token=True,
        token=HF_TOKEN
    )

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    output = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        max_new_tokens=50,
    )
    output = output[0].to("cpu")
    print(tokenizer.decode(output))

def generate_with_llama(prompt):
    model = "meta-llama/Llama-2-7b-chat-hf"

    tokenizer = AutoTokenizer.from_pretrained(model)
    text_gen = pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    sequences = text_gen(
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
        print("Choose a model to interact with:")
        print("1: Falcon")
        print("2: Llama")
        print("3: Quit")
        choice = input("Enter your choice: ")

        if choice == '1':
            user_input = input("Type your prompt for Falcon: ")
            generate_with_falcon(user_input)
        elif choice == '2':
            user_input = input("Type your prompt for Llama: ")
            generate_with_llama(user_input)
        elif choice == '3':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")
