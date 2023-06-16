import torch
from tqdm.auto import tqdm
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import StepLR
from torch.optim import Adam

def train_model(model, train_loader, val_loader, optimizer, scheduler, tokenizer, unfreeze_layers, device, n_epochs=4):
    """
    Train the model and evaluate on the validation set after each epoch.
    """

    # Move the model to the GPU
    model = model.to(device)
    loss_fct = CrossEntropyLoss()

    for epoch in range(n_epochs):  # Number of epochs
        # Training phase
        model.train()
        epoch_iterator = tqdm(train_loader, desc="Training (Epoch #{})".format(epoch))
        for batch in epoch_iterator:

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            model.zero_grad()
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # Calculate loss
            loss = loss_fct(outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1))

            # Backward pass and optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # Update the progress bar
            epoch_iterator.set_postfix(loss=loss.item())

        # Evaluation phase
        eval_loss = 0.0
        model.eval()  # Put the model in eval mode
        eval_iterator = tqdm(val_loader, desc="Evaluation (Epoch #{})".format(epoch))
        for batch in eval_iterator:
            with torch.no_grad():  # No need to calculate gradients in eval mode
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                # Get model's output
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = loss_fct(outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1))

                eval_loss += loss.item()  # Accumulate the validation loss

                # Decode output and target tokens
                for i in range(input_ids.shape[0]):  # Loop over each item in the batch
                    output_text = tokenizer.decode(outputs.logits[i].argmax(-1).tolist(), skip_special_tokens=True)
                    target_text = tokenizer.decode(labels[i].tolist(), skip_special_tokens=True)

                    scores = compute_detailed_scores_pytorch(target_text, output_text)

        # Compute average ROUGE scores and loss over the validation set
        eval_loss /= len(val_loader)

        # Print average ROUGE scores and loss
        print(f"Validation loss: {eval_loss}")
        print(scores)

        if epoch < 3:
            unfreeze_layers(model, "encoder", [-i for i in range(1, epoch + 2)])
            unfreeze_layers(model, 'decoder', [-i for i in range(1, epoch + 2)])
            optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-8, betas=(0.9, 0.999), eps=1e-8)
