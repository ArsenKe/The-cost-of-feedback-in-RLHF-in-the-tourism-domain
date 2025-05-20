import os
import gradio as gr
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from firebase_utils import FirebaseManager
from feedback_handler import FeedbackHandler 
from functools import lru_cache

# Initialize Firebase based on environment
if os.getenv('SPACE_ID'):
    if os.getenv('FIREBASE_CREDENTIALS'):
        credentials_path = 'firebase-credentials.json'
        with open(credentials_path, 'w') as f:
            f.write(os.getenv('FIREBASE_CREDENTIALS'))
        firebase_manager = FeedbackHandler(
            database_url="https://rlhf-2e2cc-default-rtdb.europe-west1.firebasedatabase.app"
        )
    else:
        raise ValueError("Firebase credentials not found in environment variables!")
# Load model & tokenizer
tokenizer = AutoTokenizer.from_pretrained("ArsenKe/MT5_large_finetuned_chatbot")
model = AutoModelForSeq2SeqLM.from_pretrained("ArsenKe/MT5_large_finetuned_chatbot")

pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, framework="pt")

# Generate chatbot responses
@lru_cache(maxsize=500)

def get_chatbot_responses(prompt):
    try:
        input_text = f"{prompt}" 
        response1 = pipe(input_text, max_length=512, temperature=0.5, do_sample=True)[0]['generated_text']
        response2 = pipe(input_text, max_length=512, temperature=0.9, do_sample=True)[0]['generated_text']
        return response1.strip(), response2.strip()
    except Exception as e:
        return f"Error: {e}", f"Error: {e}"

# Save Feedback to Firebase
def submit_feedback(prompt, response1, response2, selected_response, rating, speed_rating, 
                    inappropriate, hallucination, constraint_satisfaction, 
                    sexual_content, violent_content, encourages_violence,
                    denigrates_class, harmful_advice, expresses_opinion,
                    expresses_moral_judgment):
    feedback = {
        "prompt": prompt,
        "responses": {
            "response1": response1,
            "response2": response2,
            "selected": f"Response {selected_response}"
        },
        "ratings": {
            "overall_quality": rating,
            "response_speed": speed_rating,
            "assessment": {
                "inappropriate_content": inappropriate == "Yes",
                "hallucination": hallucination == "Yes",
                "satisfies_constraints": constraint_satisfaction == "Yes",
                "sexual_content": sexual_content == "Yes",
                "violent_content": violent_content == "Yes",
                "encourages_violence": encourages_violence == "Yes",
                "denigrates_protected_class": denigrates_class == "Yes",
                "harmful_advice": harmful_advice == "Yes",
                "expresses_opinion": expresses_opinion == "Yes",
                "expresses_moral_judgment": expresses_moral_judgment == "Yes"
            }
        },
        "metadata": {
            "model": "ArsenKe/MT5_large_finetuned_chatbot",
            "feedback_version": "1.1"
        }
    }
    try:
        success, message = firebase_manager.store_feedback(feedback)
        return message
    except Exception as e:
        return f"Error submitting feedback: {str(e)}"
# Create Gradio Interface
with gr.Blocks(theme=gr.themes.Soft()) as interface:
    gr.Markdown(" RLHF Chatbot for Tourism Assistance")
    
    # Instruction Section
    with gr.Accordion("Instructions (Click to Expand)", open=False):
        gr.Markdown("""
        - Type a **question or request** in the input box.
        - The chatbot will generate **two responses**.
        - **Select the better response** and **rate its quality**.
        - Provide additional feedback if necessary.
        """)

    #  User Input Section
    with gr.Row():
        prompt_input = gr.Textbox(label="User Prompt", lines=2, placeholder="Enter your query...")  
        generate_button = gr.Button("💬 Generate Responses", variant="primary")

    #  Chatbot Responses
    with gr.Row():
        chatbot_response1 = gr.Textbox(label="Chatbot Response 1", interactive=False)
        chatbot_response2 = gr.Textbox(label="Chatbot Response 2", interactive=False)

    #  Selection & Rating
    with gr.Row():
        selected_response = gr.Radio(choices=["1", "2"], label="Which response is better?", interactive=True)
        rating = gr.Slider(minimum=1, maximum=5, step=1, label="⭐ Rate the response quality")
        speed_rating = gr.Slider(minimum=1, maximum=5, step=1, label="⚡ Rate the response speed")

    #  Additional Feedback (Collapsible)
    with gr.Accordion("🛠 Advanced Assessment ", open=False):
        with gr.Row():
            with gr.Column():
                inappropriate = gr.Radio(choices=["Yes", "No"], label="Inappropriate for customer?", value="No")
                hallucination = gr.Radio(choices=["Yes", "No"], label="Contains hallucination?", value="No")
                constraint_satisfaction = gr.Radio(choices=["Yes", "No"], label="Satisfies constraints?", value="Yes")
                sexual_content = gr.Radio(choices=["Yes", "No"], label="Contains sexual content?", value="No")
                violent_content = gr.Radio(choices=["Yes", "No"], label="Contains violent content?", value="No")
            with gr.Column():
                encourages_violence = gr.Radio(choices=["Yes", "No"], label="Encourages violence?", value="No")
                denigrates_class = gr.Radio(choices=["Yes", "No"], label="Denigrates protected class?", value="No")
                harmful_advice = gr.Radio(choices=["Yes", "No"], label="Gives harmful advice?", value="No")
                expresses_opinion = gr.Radio(choices=["Yes", "No"], label="Expresses opinion?", value="No")
                expresses_moral_judgment = gr.Radio(choices=["Yes", "No"], label="Expresses moral judgment?", value="No")


    #  Feedback Submission
    feedback_status = gr.Textbox(label="Submission Status", interactive=False)
    with gr.Row():
        submit_button = gr.Button("✅ Submit Feedback", variant="primary")

    #  Event Handlers
    generate_button.click(fn=get_chatbot_responses, inputs=[prompt_input], outputs=[chatbot_response1, chatbot_response2])
    submit_button.click(
        fn=submit_feedback,
        inputs=[
            prompt_input, chatbot_response1, chatbot_response2,
            selected_response, rating, speed_rating,
            inappropriate, hallucination, constraint_satisfaction,
            sexual_content, violent_content, encourages_violence,
            denigrates_class, harmful_advice, expresses_opinion,
            expresses_moral_judgment
        ],
        outputs=[feedback_status]
    )

#if __name__ == "__main__":
 #  interface.launch(share=True, server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    # Check if running on Spaces
    if os.getenv('SPACE_ID'):
        interface.launch()
    else:
        # Local development settings
        interface.launch(share=True, server_name="0.0.0.0", server_port=7860)