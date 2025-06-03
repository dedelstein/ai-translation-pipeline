import random
from src.process_data import translate_text

MAX_LEN_MULTIPLIER = 3  # if translated is 3x longer than Finnish
MIN_FINNISH_LEN_FOR_CHECK = 3

DEFAULT_PERTURBATION_TOKENS = ["<", ">", "&", " !", " ?", "/", " -"]


def filter_results(raw_results):
    filtered_results = []
    finnish_text_segments = []  # For cleaner list building

    if raw_results:  # Check if raw_results is not None or empty
        for region_info in raw_results:
            text_content = region_info[1][0]
            confidence = region_info[1][1]

            is_likely_text = True
            if len(text_content.strip()) == 0:  # Empty or whitespace only
                is_likely_text = False
            elif not any(
                char.isalpha() for char in text_content
            ):  # Single char and not alphanumeric
                print(
                    f"DEBUG: Filtering out OCR result with no letters: '{text_content}'"
                )
                is_likely_text = False
            elif len(text_content) <= 1:
                print(f"DEBUG: Filtering out short OCR result: '{text_content}'")
                is_likely_text = False
            if confidence < 0.9:
                print(
                    f"DEBUG: Filtering out low conf OCR result: '{text_content}, {confidence}'"
                )
                is_likely_text = False

            if is_likely_text:
                filtered_results.append(region_info)
                finnish_text_segments.append(text_content)

    return filtered_results, finnish_text_segments


def attempt_retranslation_with_perturbation(
    finnish_text_segment,
    translator_tokenizer,
    translator_model,
    perturbation_options=DEFAULT_PERTURBATION_TOKENS,
):
    """
    Attempts to re-translate by adding a randomly chosen perturbation token
    to a randomly chosen position (prefix or suffix).
    """
    if not perturbation_options:
        return finnish_text_segment

    # 1. Choose a random perturbation token
    perturbation_token = random.choice(perturbation_options)

    # 2. Choose a random position
    position = random.choice(["prefix", "suffix"])

    # 3. Construct the perturbed input based on the chosen position
    if position == "prefix":
        perturbed_input = perturbation_token + finnish_text_segment
    else:  # suffix
        perturbed_input = finnish_text_segment + perturbation_token

    print(
        f"DEBUG: Perturbing '{finnish_text_segment}' with token '{perturbation_token}' at position '{position}'"
    )

    decoding_params = {"temperature": 1.5, "repetition_penalty": 2.0}

    # Assuming translate_text is defined elsewhere
    perturbed_translation_list = translate_text(
        [perturbed_input], translator_tokenizer, translator_model, decoding_params
    )
    perturbed_translation = (
        perturbed_translation_list[0] if perturbed_translation_list else ""
    )

    # 4. Update cleanup logic to check both prefix and suffix
    if perturbation_token and perturbed_translation.endswith(perturbation_token):
        # If the token is at the end of the output, strip it from the end
        perturbed_translation = perturbed_translation[
            : -len(perturbation_token)
        ].strip()
    elif perturbation_token and perturbed_translation.startswith(perturbation_token):
        # If the token is at the beginning of the output, strip it from the beginning
        perturbed_translation = perturbed_translation[len(perturbation_token) :].strip()

    return perturbed_translation


def clean_translations(
    finnish_text, translated_text, translator_tokenizer, translator_model
):
    final_corrected_translations = []
    for i, fi_segment in enumerate(finnish_text):
        en_segment = translated_text[i]
        is_problematic = False

        # Check 1: Excessive length
        if (
            len(fi_segment) >= MIN_FINNISH_LEN_FOR_CHECK
            and len(en_segment) > len(fi_segment) * MAX_LEN_MULTIPLIER
        ):
            is_problematic = True
            print(
                f"DEBUG: Segment '{fi_segment}' -> excessively long translation detected: '{en_segment[:100]}...'"
            )

        # Check 2: Repetition (simple heuristic)
        if not is_problematic:
            words = en_segment.split()
            if len(words) > 5:
                counts = {}
                for word in words[:10]:
                    counts[word] = counts.get(word, 0) + 1
                if any(count > 3 for count in counts.values()):
                    is_problematic = True
                    print(
                        f"DEBUG: Segment '{fi_segment}' -> repetitive translation detected: '{en_segment[:100]}...'"
                    )

        if is_problematic:
            # Attempt re-translation with perturbation
            print(
                f"INFO: Problematic translation for '{fi_segment}'. Attempting re-translation."
            )

            corrected_segment = attempt_retranslation_with_perturbation(
                fi_segment, translator_tokenizer, translator_model
            )
            print(f"DEBUG: Re-translated '{fi_segment}' to '{corrected_segment}'")
            if len(corrected_segment) > len(fi_segment) * MAX_LEN_MULTIPLIER:
                print(
                    f"DEBUG: Re-translatation failed due to length heuristic, returning {fi_segment}"
                )
                corrected_segment = fi_segment

            final_corrected_translations.append(corrected_segment)
        else:
            final_corrected_translations.append(en_segment)

    return final_corrected_translations
