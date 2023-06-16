from torchmetrics.text import ROUGEScore

def unfreeze_layers(model, part, list_layers):
    """
    Unfreeze certain layers of the model for fine-tuning
    """
    model.model.shared.weight.requires_grad = True
    model.model.encoder.embed_positions.weight.requires_grad = True

    if part == "encoder":
        for layer in list_layers:
            for param in model.model.encoder.layers[layer].parameters():
                param.requires_grad = True

            for param in model.model.encoder.layernorm_embedding.parameters():
                param.requires_grad = True

            for param in model.model.encoder.layer_norm.parameters():
                param.requires_grad = True

    elif part == "decoder":
        for layer in list_layers:
            for param in model.model.decoder.layers[layer].parameters():
                param.requires_grad = True

            for param in model.model.decoder.layernorm_embedding.parameters():
                param.requires_grad = True

            for param in model.model.decoder.layer_norm.parameters():
                param.requires_grad = True
    else:
        raise ValueError("Invalid part. Choose 'encoder' or 'decoder'")

def compute_detailed_scores_pytorch(target_text, output_text):
    """
    Compute the evaluation metrics
    """
    # Calculate ROUGE score
    rouge = ROUGEScore()
    rouge_score = rouge(output_text, target_text)

    # Return the scores
    return {'rouge': rouge_score}
