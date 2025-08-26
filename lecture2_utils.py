import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Import necessary libraries for the interactive plot and calculations
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider
from IPython.display import display, HTML
from ipywidgets import interactive, Text

import numpy as np
import re
from gensim.models import KeyedVectors
from sklearn.decomposition import PCA
import warnings
import gensim.downloader as api

warnings.filterwarnings("ignore")  # Ignore warnings for cleaner output


def cos_similarity(emb1, emb2):
    dot_product = np.dot(emb1, emb2)
    norm_result = np.linalg.norm(emb1)
    norm_comparison = np.linalg.norm(emb2)
    similarity_score = -1
    if norm_result != 0 and norm_comparison != 0:
        similarity_score = dot_product / (norm_result * norm_comparison)
    return similarity_score


def get_vector_results(expression, comparison_word):
    """
    Performs word vector arithmetic and returns the 2D vector positions
    and cosine similarity.

    Args:
        expression (str): A word expression (e.g., "king - man + woman").
        comparison_word (str): A single word to compare the result to.

    Returns:
        dict: A dictionary containing:
            - "expression": The input expression string.
            - "comparison_word": The input comparison word.
            - "vector_positions": A dictionary mapping each word to its 2D coordinates.
            - "similarity": The cosine similarity score between the result and comparison vectors.
    """
    # --- 1. Load a smaller word vector model (GloVe) ---
    MODEL_NAME = "glove-wiki-gigaword-50"
    try:
        word2vec_model = api.load(MODEL_NAME)
    except Exception as e:
        print(f"Error loading model: {e}")
        return {}

    # --- 2. Process inputs and perform vector arithmetic ---
    expression = expression.strip().lower()
    comparison_word = comparison_word.strip().lower()

    if not expression:
        print("Please provide a valid expression.")
        return {}

    tokens = re.findall(r"(\w+|-|\+)", expression.replace(" ", ""))

    if not tokens or tokens[0] in ("+", "-"):
        print("Invalid expression format. Start with a word.")
        return {}

    all_words = [t for t in tokens if t not in ("+", "-")]
    if comparison_word:
        all_words.append(comparison_word)

    try:
        for word in all_words:
            _ = word2vec_model.get_vector(word)
    except KeyError as e:
        print(f"The word {e} is not in the model's vocabulary.")
        return {}

    # Get the vector for the first word and make it writeable
    result_vector = word2vec_model.get_vector(tokens[0]).copy()
    individual_vectors = {tokens[0]: word2vec_model.get_vector(tokens[0])}

    i = 1
    while i < len(tokens):
        operator = tokens[i]
        word = tokens[i + 1]
        word_vector = word2vec_model.get_vector(word)
        individual_vectors[word] = word_vector
        if operator == "+":
            result_vector += word_vector
        elif operator == "-":
            result_vector -= word_vector
        i += 2

    comparison_vector = None
    if comparison_word:
        comparison_vector = word2vec_model.get_vector(comparison_word)

    # --- 3. Dimensionality Reduction for 2D position ---
    all_vectors = list(individual_vectors.values()) + [result_vector]
    if comparison_vector is not None:
        all_vectors.append(comparison_vector)

    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform(all_vectors)

    all_words_for_display = list(individual_vectors.keys()) + ["Result"]
    if comparison_word:
        all_words_for_display.append(comparison_word)

    vector_map_2d = {
        word: list(reduced_vectors[i]) for i, word in enumerate(all_words_for_display)
    }

    # --- 4. Calculate Cosine Similarity Manually ---
    similarity_score = -1
    if comparison_vector is not None:
        dot_product = np.dot(result_vector, comparison_vector)
        norm_result = np.linalg.norm(result_vector)
        norm_comparison = np.linalg.norm(comparison_vector)
        if norm_result != 0 and norm_comparison != 0:
            similarity_score = dot_product / (norm_result * norm_comparison)

    # --- Plotting the vectors ---
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot individual vectors (expression words)
    for word in individual_vectors.keys():
        coords = vector_map_2d[word]
        ax.arrow(
            0,
            0,
            coords[0],
            coords[1],
            head_width=0.03,
            head_length=0.03,
            fc="blue",
            ec="blue",
        )
        ax.text(coords[0] * 1.1, coords[1] * 1.1, word, fontsize=12, color="blue")

    # Plot the calculated result vector
    result_coords = vector_map_2d["Result"]
    ax.arrow(
        0,
        0,
        result_coords[0],
        result_coords[1],
        head_width=0.05,
        head_length=0.05,
        fc="red",
        ec="red",
        linewidth=2,
        linestyle="--",
    )
    ax.text(
        result_coords[0] * 1.1,
        result_coords[1] * 1.1,
        f"Result",
        fontsize=14,
        color="red",
        fontweight="bold",
    )

    # Plot the comparison vector if provided
    if comparison_word:
        comparison_coords = vector_map_2d[comparison_word]
        ax.arrow(
            0,
            0,
            comparison_coords[0],
            comparison_coords[1],
            head_width=0.05,
            head_length=0.05,
            fc="green",
            ec="green",
            linewidth=2,
            linestyle="-",
        )
        ax.text(
            comparison_coords[0] * 1.1,
            comparison_coords[1] * 1.1,
            f"Compare: '{comparison_word}'",
            fontsize=14,
            color="green",
            fontweight="bold",
        )

    # Add a title and labels
    ax.set_title(f"Word2Vec Arithmetic: {expression}", fontsize=16)
    ax.set_xlabel("PCA Dimension 1", fontsize=12)
    ax.set_ylabel("PCA Dimension 2", fontsize=12)

    # Set plot limits
    all_x = [v[0] for v in reduced_vectors]
    all_y = [v[1] for v in reduced_vectors]
    x_max, x_min = np.max(all_x) * 1.2, np.min(all_x) * 1.2
    y_max, y_min = np.max(all_y) * 1.2, np.min(all_y) * 1.2
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Add origin lines
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)

    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    plt.show()

    return {
        "expression": expression,
        "comparison_word": comparison_word,
        "vector_positions": vector_map_2d,
        "similarity": similarity_score,
    }


# --- 1. Define the Softmax function with temperature scaling ---
def softmax_with_temperature(logits, T):
    """
    Applies the softmax function with a temperature parameter T.

    Args:
        logits (list): A list of raw scores from a model.
        T (float): The temperature parameter.

    Returns:
        np.array: A numpy array of probabilities.
    """
    # Softmax becomes unstable for T close to zero, so we handle that case
    if T <= 0:
        T = 1e-9  # A very small number to prevent division by zero

    # Calculate the exponentiated logits
    exponentiated_logits = np.exp(np.array(logits) / T)

    # Calculate the sum of exponentiated logits
    sum_exp_logits = np.sum(exponentiated_logits)

    # Return the normalized probabilities
    return exponentiated_logits / sum_exp_logits


# --- 2. Define the plotting function that will be linked to the slider ---
def update_plot(temperature):
    """
    Generates a bar plot showing the probabilities for the given temperature.

    Args:
        temperature (float): The value from the interactive slider.
    """
    # We first calculate the corresponding logits by taking the log of the probabilities.
    # This assumes the original probabilities were calculated with a temperature of 1.0.
    initial_data = [
        {"token": " are", "id": 389, "probability": 0.09539596736431122},
        {"token": " have", "id": 423, "probability": 0.07926966995000839},
        {"token": " will", "id": 481, "probability": 0.0715038999915123},
        {"token": ",", "id": 11, "probability": 0.065525121986866},
        {"token": " (", "id": 357, "probability": 0.029835928231477737},
    ]

    # Extract the labels and initial probabilities
    labels = [item["token"] for item in initial_data]
    initial_probabilities = [item["probability"] for item in initial_data]

    # Calculate the logits from the initial probabilities
    logits = np.log(initial_probabilities)

    # Calculate the probabilities for the current temperature using the logits
    probabilities = softmax_with_temperature(logits, temperature)

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 5))

    # The labels for our tokens on the x-axis
    x_positions = np.arange(len(labels))

    # Create the bar chart
    ax.bar(
        x_positions,
        probabilities,
        color=["skyblue", "salmon", "lightgreen", "orange", "lightgray"],
    )

    # Set the x-axis labels
    plt.xticks(x_positions, labels)

    # Add titles and labels
    ax.set_title(
        f"Probability Distribution (Temperature = {temperature:.2f})", fontsize=16
    )
    ax.set_xlabel("Tokens", fontsize=12)
    ax.set_ylabel("Probability", fontsize=12)
    ax.set_ylim(0, 1.0)  # Ensure y-axis scale is consistent

    # Display the probability value on top of each bar
    for i, prob in enumerate(probabilities):
        ax.text(i, prob + 0.02, f"{prob:.2f}", ha="center", fontsize=12)

    # Add a horizontal line at 1/5 (0.2) to visualize a uniform distribution
    ax.axhline(1 / len(labels), color="gray", linestyle="--", linewidth=1)

    # Clean up and show the plot
    plt.tight_layout()
    plt.show()


def visualize_tokenization(sentence):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    """
    Takes a sentence, tokenizes it, and displays the tokens and their IDs in a
    formatted HTML table.

    Args:
        sentence (str): The input sentence from the user.
    """
    if not sentence:
        display(
            HTML(
                "<div><p class='text-gray-500'>Please enter a sentence above.</p></div>"
            )
        )
        return

    # Tokenize the input sentence
    # We use `return_tensors=None` to get a list of integers (token IDs)
    input_ids = tokenizer.encode(
        sentence, return_tensors=None, add_special_tokens=False
    )

    # Convert the token IDs back into the actual token strings for display
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # Build the HTML output to display the results in a clean table
    html_output = f"""
    <div style="font-family: 'Inter', sans-serif; padding: 20px; border: 1px solid #e5e7eb; border-radius: 8px; background-color: #f9fafb;">
        <h3 style="font-size: 1.25rem; font-weight: 600; margin-bottom: 12px; color: #1f2937;">Tokenization Breakdown</h3>
        <p style="font-size: 1rem; color: #4b5563; margin-bottom: 16px;">
            Input Sentence: <strong>"{sentence}"</strong>
        </p>
        <table style="width: 100%; border-collapse: collapse;">
            <thead style="background-color: #f3f4f6; color: #4b5563;">
                <tr>
                    <th style="padding: 12px; text-align: left; border-bottom: 2px solid #e5e7eb;">Token</th>
                    <th style="padding: 12px; text-align: left; border-bottom: 2px solid #e5e7eb;">Token ID</th>
                </tr>
            </thead>
            <tbody>
    """

    # Add a row for each token and its ID
    for token, token_id in zip(tokens, input_ids):
        # We replace the special character 'Ġ' with a space for better readability
        display_token = token.replace("Ġ", " ").strip()
        html_output += f"""
            <tr style="border-bottom: 1px solid #e5e7eb;">
                <td style="padding: 12px; font-weight: 500; color: #1f2937;">'{display_token}'</td>
                <td style="padding: 12px; font-family: monospace; color: #374151;">{token_id}</td>
            </tr>
        """

    html_output += """
            </tbody>
        </table>
    </div>
    """

    display(HTML(html_output))
