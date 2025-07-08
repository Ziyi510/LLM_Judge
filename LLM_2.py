from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import torch
# import flash_attn
import json
import re
import time
import os

model_name_qwen = "Qwen/Qwen3-8B"
model_name_llama = "meta-llama/Meta-Llama-3.1-8B-Instruct"
model_name_mistral = "mistralai/Mistral-7B-Instruct-v0.3"
model_name_gemma = "google/gemma-7b-it"

tokenizer_qwen = AutoTokenizer.from_pretrained(
    model_name_qwen,
    trust_remote_code = True,
    cache_dir="/scratch"
)

model_qwen = AutoModelForCausalLM.from_pretrained(
    model_name_qwen,
    cache_dir="/scratch",
    torch_dtype = torch.bfloat16,
    # use_flash_attention_2=True,
    # device_map = "auto",
    device_map = {"": "cuda:0"},  # Use GPU 0 for Qwen
)

tokenizer_llama = AutoTokenizer.from_pretrained(
    model_name_llama,
    trust_remote_code = True,
    cache_dir="/scratch"
)

model_llama = AutoModelForCausalLM.from_pretrained(
    model_name_llama,
    cache_dir="/scratch",
    torch_dtype = torch.bfloat16,
    # use_flash_attention_2=True,
    # device_map = "auto",
    device_map = {"": "cuda:1"},  # Use GPU 1 for Llama
)

tokenizer_mistral = AutoTokenizer.from_pretrained(
    model_name_mistral,
    trust_remote_code=True,
    cache_dir="/scratch"
)

model_mistral = AutoModelForCausalLM.from_pretrained(
    model_name_mistral,
    cache_dir="/scratch",
    torch_dtype=torch.bfloat16,
    # use_flash_attention_2=True,
    # device_map="auto",
    device_map = {"": "cuda:2"},  # Use GPU 2 for Mistral
)

tokenizer_gemma = AutoTokenizer.from_pretrained(
    model_name_gemma,
    trust_remote_code=True,
    cache_dir="/scratch"
)

model_gemma = AutoModelForCausalLM.from_pretrained(
    model_name_gemma,
    cache_dir="/scratch",
    torch_dtype=torch.bfloat16,
    device_map = {"": "cuda:3"},  # Use GPU 3 for Gemma
)

dataset = load_dataset("GBaker/MedQA-USMLE-4-options")
train_data = dataset['train']

def format_options(opt_dict):
    return "\n".join([f"{k}. {v}" for k, v in opt_dict.items()])

def generate_answer_and_explanation(model, tokenizer, question, options, model_name="Qwen"):
    """
    Generate answer choice and explanation for a given question
    """
    prompt = f"""Question: {question}
Options:
{format_options(options)}

You are a helpful medical assistant. select the BEST answer and generate the choice (A/B/C/D), then explain your reasoning.
Follow the format strictly:
<Choice>. Explanation: <your explanation here>
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode only the generated part (exclude the input prompt)
    generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    # Parse the response to extract answer choice and explanation
    answer_choice = None
    explanation = ""
    
    # Look for answer choice (A, B, C, or D)
    answer_match = re.search(r'\b([ABCD])\b', generated_text, re.IGNORECASE)
    if answer_match:
        answer_choice = answer_match.group(1).upper()
    
    # Extract explanation (everything after the answer choice)
    if answer_choice:
        explanation_start = generated_text.find(answer_choice) + 1
        explanation = generated_text[explanation_start:].strip()
        # Remove common prefixes
        explanation = re.sub(r'^[.:\s]+', '', explanation)
    
    return {
        'answer_choice': answer_choice,
        'explanation': explanation,
        'full_response': generated_text
    }

def evaluate_answer_explanation(model, tokenizer, question, options, correct_answer, predicted_answer, explanation, model_name="Llama"):
    """
    Evaluate the quality of answer and explanation on a 1-3 scale, and provide a confidence score (1-5)
    """
    evaluation_prompt = f"""You are a helpful medical assistant and expert evaluator. 
Strictly follow the evaluation rules below to rate the following answer and explanation for a medical question:

Evaluation rules:
1. If the predicted answer is NOT the same as the correct answer, ALWAYS give a rating of 1, regardless of the explanation.
2. If the predicted answer is correct BUT the explanation is wrong, incomplete, or does NOT logically support the answer, give a rating of 2.
3. Only if BOTH the predicted answer is correct AND the explanation is clear, detailed, and accurately supports the answer, give a rating of 3.

Do NOT give a rating of 2 or 3 to any answer that is not correct, even if the explanation seems plausible.
If you are unsure, be conservative and use a lower rating.

In addition, provide your confidence in your rating on a scale of 1 (lowest confidence) to 5 (highest confidence).

Question: {question}
Options:
{format_options(options)}

Predicted Answer: {predicted_answer}
Explanation: {explanation}

Please provide:
1. A rating from 1-3
2. Your confidence in your rating (1-5)
3. A brief justification for your rating

Format your answer as:
Rating: <number>
Confidence: <number>
Justification: <text>
"""

    inputs = tokenizer(evaluation_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode only the generated part
    generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    # Extract rating, justification, and confidence
    rating = None
    justification = ""
    confidence = None
    
    # Look for 'Rating: <number>'
    rating_match = re.search(r'Rating[:\s]*([1-3])', generated_text)
    if rating_match:
        rating = int(rating_match.group(1))
    
    # Look for 'Justification: <text>'
    justification_match = re.search(r'Justification[:\s]*(.*?)(?:Confidence:|$)', generated_text, re.DOTALL)
    if justification_match:
        justification = justification_match.group(1).strip()
    
    # Look for 'Confidence: <number>'
    confidence_match = re.search(r'Confidence[:\s]*([1-5])', generated_text)
    if confidence_match:
        confidence = int(confidence_match.group(1))
    
    return {
        'rating': rating,
        'justification': justification,
        'confidence': confidence,
        'full_evaluation': generated_text
    }

def process_train_dataset_with_evaluation(qwen_model, qwen_tokenizer, llama_model, llama_tokenizer, train_data, max_samples=None):
    """
    Process the entire train dataset with Qwen generating answers and Llama evaluating them
    """
    results = []
    
    # Limit samples if specified (useful for testing)
    if max_samples:
        train_data = train_data.select(range(min(max_samples, len(train_data))))
    
    print(f"Processing {len(train_data)} samples with Qwen + Llama evaluation...")
    
    for i, sample in enumerate(train_data):
        print(f"Processing sample {i+1}/{len(train_data)}")
        
        question = sample['question']
        options = sample['options']
        correct_answer = sample.get('answer', None)
        
        try:
            # Step 1: Generate answer and explanation with Qwen
            print("  Generating answer with Qwen...")
            qwen_result = generate_answer_and_explanation(qwen_model, qwen_tokenizer, question, options, "Qwen")
            
            # Step 2: Evaluate the answer-explanation pair with Llama
            print("  Evaluating with Llama...")
            evaluation_result = evaluate_answer_explanation(
                llama_model, 
                llama_tokenizer, 
                question, 
                options, 
                correct_answer, 
                qwen_result['answer_choice'], 
                qwen_result['explanation'], 
                "Llama"
            )
            
            # Store comprehensive results
            sample_result = {
                'sample_id': i,
                'question': question,
                'options': options,
                'correct_answer': correct_answer,
                'qwen_predicted_answer': qwen_result['answer_choice'],
                'qwen_explanation': qwen_result['explanation'],
                'qwen_full_response': qwen_result['full_response'],
                'llama_rating': evaluation_result['rating'],
                'llama_justification': evaluation_result['justification'],
                'llama_full_evaluation': evaluation_result['full_evaluation'],
                'is_correct': qwen_result['answer_choice'] == correct_answer if correct_answer else None
            }
            
            results.append(sample_result)
            
            # Print progress
            print(f"  Qwen Answer: {qwen_result['answer_choice']}, Correct: {correct_answer}")
            print(f"  Llama Rating: {evaluation_result['rating']}/3")
            if evaluation_result['justification']:
                print(f"  Justification: {evaluation_result['justification'][:100]}...")
            print()
            
            # Add a small delay to avoid overwhelming the system
            time.sleep(0.2)
            
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            results.append({
                'sample_id': i,
                'question': question,
                'options': options,
                'correct_answer': correct_answer,
                'qwen_predicted_answer': None,
                'qwen_explanation': None,
                'qwen_full_response': None,
                'llama_rating': None,
                'llama_justification': None,
                'llama_full_evaluation': None,
                'is_correct': None,
                'error': str(e)
            })
    
    return results

def calculate_comprehensive_metrics(results):
    """
    Calculate comprehensive metrics including accuracy and average rating
    """
    # Accuracy metrics
    correct_predictions = sum(1 for r in results if r.get('is_correct') == True)
    total_predictions = sum(1 for r in results if r.get('is_correct') is not None)
    
    # Rating metrics
    valid_ratings = [r.get('llama_rating') for r in results if r.get('llama_rating') is not None]
    average_rating = sum(valid_ratings) / len(valid_ratings) if valid_ratings else None
    
    # Rating distribution
    rating_distribution = {}
    for rating in valid_ratings:
        rating_distribution[rating] = rating_distribution.get(rating, 0) + 1
    
    print("=== COMPREHENSIVE METRICS ===")
    if total_predictions > 0:
        accuracy = correct_predictions / total_predictions
        print(f"Accuracy: {accuracy:.4f} ({correct_predictions}/{total_predictions})")
    
    if average_rating is not None:
        print(f"Average Llama Rating: {average_rating:.2f}/3")
        print("Rating Distribution:")
        for rating in sorted(rating_distribution.keys()):
            count = rating_distribution[rating]
            percentage = (count / len(valid_ratings)) * 100
            print(f"  {rating}/3: {count} samples ({percentage:.1f}%)")
    
    return {
        'accuracy': accuracy if total_predictions > 0 else None,
        'average_rating': average_rating,
        'rating_distribution': rating_distribution
    }

def save_results(results, filename):
    """
    Save results to a JSON file
    """
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {filename}")

def generate_answers_all_models(models, tokenizers, question, options, model_names):
    """
    Generate answers and explanations for a question using all provided models.
    """
    results = {}
    for model, tokenizer, name in zip(models, tokenizers, model_names):
        result = generate_answer_and_explanation(model, tokenizer, question, options, model_name=name)
        results[name] = result
    return results

def cross_review_answers(models, tokenizers, model_names, question, options, correct_answer, all_model_results):
    """
    Each model reviews the other models' answer+explanation.
    Returns a dict: {submitter: {reviewer: review_result, ...}, ...}
    """
    reviews = {}
    for i, (submitter_name, submitter_result) in enumerate(all_model_results.items()):
        reviews[submitter_name] = {}
        for j, (reviewer_model, reviewer_tokenizer, reviewer_name) in enumerate(zip(models, tokenizers, model_names)):
            if reviewer_name == submitter_name:
                continue  # Skip self-review
            review = evaluate_answer_explanation(
                reviewer_model,
                reviewer_tokenizer,
                question,
                options,
                correct_answer,
                submitter_result['answer_choice'],
                submitter_result['explanation'],
                reviewer_name
            )
            reviews[submitter_name][reviewer_name] = review
    return reviews

def process_dataset_with_cross_reviews(models, tokenizers, model_names, data, max_samples=None):
    """
    For each sample, generate answers from all models and have each model review the others.
    Returns a list of results with answers, cross-reviews, and Gemma's contextual answer.
    """
    results = []
    if max_samples:
        data = data.select(range(min(max_samples, len(data))))
    print(f"Processing {len(data)} samples with cross-model answer generation and review...")
    for i, sample in enumerate(data):
        print(f"Sample {i+1}/{len(data)}")
        question = sample['question']
        options = sample['options']
        correct_answer = sample.get('answer', None)
        # Step 1: All models generate answers
        all_model_results = generate_answers_all_models(models, tokenizers, question, options, model_names)
        # Step 2: Cross-review
        cross_reviews = cross_review_answers(models, tokenizers, model_names, question, options, correct_answer, all_model_results)
        # Step 3: Gemma generates answer with context
        gemma_result = generate_gemma_with_context(model_gemma, tokenizer_gemma, question, options, all_model_results, cross_reviews)
        # Store everything in the required format
        results.append({
            'sample_id': i,
            'question': question,
            'options': options,
            'correct_answer': correct_answer,
            'model_answers': all_model_results,  # ans+explanation
            'cross_reviews': cross_reviews,      # peer review
            'gemma_result': gemma_result         # gemma result
        })
        print(f"  Done sample {i+1}")
        time.sleep(0.2)
    return results

def generate_gemma_with_context(model_gemma, tokenizer_gemma, question, options, all_model_results, cross_reviews):
    """
    Use Gemma to generate an answer+explanation, given the question, options, all previous model answers+explanations, and all reviews+confidences. Do NOT provide the correct answer.
    """
    # Format previous answers and reviews
    answers_section = ""
    for model_name, result in all_model_results.items():
        answers_section += f"Model: {model_name}\nAnswer: {result['answer_choice']}\nExplanation: {result['explanation']}\n\n"
    reviews_section = ""
    for submitter, reviewers in cross_reviews.items():
        for reviewer, review in reviewers.items():
            reviews_section += f"Reviewer: {reviewer} reviewing {submitter}\nRating: {review.get('rating')}\nConfidence: {review.get('confidence')}\nJustification: {review.get('justification')}\n\n"
    prompt = (
        "You are a helpful medical assistant.\n"
        "Here is a question and its options.\n"
        f"Question: {question}\n"
        f"Options:\n{format_options(options)}\n"
        "\nPrevious Model Answers and Explanations:\n"
        f"{answers_section}"
        "\nReviews of Answers:\n"
        f"{reviews_section}"
        "\nBased on all the above information, select the BEST answer and generate the choice (A/B/C/D), then explain your reasoning.\n"
        "Follow the format strictly:\n<Choice>. Explanation: <your explanation here>\n"
    )
    inputs = tokenizer_gemma(prompt, return_tensors="pt").to(model_gemma.device)
    with torch.no_grad():
        outputs = model_gemma.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            eos_token_id=tokenizer_gemma.eos_token_id,
            pad_token_id=tokenizer_gemma.eos_token_id
        )
    generated_text = tokenizer_gemma.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    answer_choice = None
    explanation = ""
    answer_match = re.search(r'\b([ABCD])\b', generated_text, re.IGNORECASE)
    if answer_match:
        answer_choice = answer_match.group(1).upper()
    if answer_choice:
        explanation_start = generated_text.find(answer_choice) + 1
        explanation = generated_text[explanation_start:].strip()
        explanation = re.sub(r'^[.:\s]+', '', explanation)
    return {
        'answer_choice': answer_choice,
        'explanation': explanation,
        'full_response': generated_text
    }

# print("Starting validation dataset processing with Qwen + llama evaluation...")
# comprehensive_results = process_train_dataset_with_evaluation(
#     model_qwen, 
#     tokenizer_qwen, 
#     model_llama, 
#     tokenizer_llama, 
#     train_data, 
#     max_samples=5  # Start with 5 samples for testing
# )
# save_results(comprehensive_results, "qwen_llama_evaluation_results.json")
# metrics = calculate_comprehensive_metrics(comprehensive_results)

# Prepare models, tokenizers, and names
models = [model_qwen, model_llama, model_mistral]
tokenizers = [tokenizer_qwen, tokenizer_llama, tokenizer_mistral]
model_names = ["Qwen", "Llama", "Mistral"]

# Process the dataset (e.g., first 5 samples for testing)
cross_review_results = process_dataset_with_cross_reviews(
    models, tokenizers, model_names, train_data, max_samples=5
)

# Save results
save_results(cross_review_results, "cross_review_results.json")
