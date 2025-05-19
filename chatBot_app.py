import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

def generate_answer(question, model):
    prompt = "Answer the following question: " + question
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(
        input_ids, 
        max_length=300,  # Adjusted max_length for shorter responses
        do_sample=True, 
        top_k=50, 
        top_p=0.9, 
        temperature=0.8,  # Slightly higher temperature for diverse output
        repetition_penalty=1.2  # Penalty to reduce repetition
    )[0]
    answer = tokenizer.decode(output, skip_special_tokens=True)
    return answer

def chatbot(question):
    answer = generate_answer(question, llm_model)
    return answer

if __name__ == "__main__":
    # Load the GPT-2 model
    llm_model = AutoModelForCausalLM.from_pretrained("gpt2")

    interface = gr.Interface(
        fn=chatbot,
        inputs="text",
        outputs="text",
        title="I am your AI Health Assistance 🏥",
        description="Ask general health-related questions to the AI Bot."
    )
    interface.launch()
