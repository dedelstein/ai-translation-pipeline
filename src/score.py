import sacrebleu
import jiwer
import string


def score_ocr(img_path, extracted_finnish_texts, ocr_ground_truth_data):

    img_key = img_path.replace("./data/", "")
    ground_truth_segments = ocr_ground_truth_data[img_key]

    hypothesis_text = " ".join(extracted_finnish_texts)
    reference_text = " ".join(ground_truth_segments)

    transform = jiwer.Compose([
                jiwer.ToLowerCase(),
                jiwer.RemoveMultipleSpaces(),
                jiwer.Strip(),
            ])
    
    transformed_reference = transform(reference_text)
    transformed_hypothesis = transform(hypothesis_text)
    # Remove punctuation for testing jiwer
    no_punct_reference = transformed_reference.translate(str.maketrans('', '', string.punctuation))
    no_punct_hypothesis = transformed_hypothesis.translate(str.maketrans('', '', string.punctuation))

    wer = jiwer.wer(no_punct_reference, no_punct_hypothesis)
    cer = jiwer.cer(no_punct_reference, no_punct_hypothesis) # CER can also use these

    return {"CER": cer, "WER": wer}

def score_translation(img_path, translated_text, deepl_translations):

        img_key = img_path.replace("./data/", "")
        ground_truth_segments = deepl_translations[img_key]

        hypothesis_full_text = " ".join(translated_text)
        reference_full_text = " ".join(ground_truth_segments)

        hypothesis_list = [hypothesis_full_text]
        reference_lists = [[reference_full_text]]

        bleu_score_details = sacrebleu.corpus_bleu(hypothesis_list, reference_lists)
        bleu_score = bleu_score_details.score

        return bleu_score