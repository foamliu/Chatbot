import random
import time

from torch import optim

from data_gen import ChatbotDataset
from models import EncoderRNN, LuongAttnDecoderRNN
from utils import *


def calc_loss(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder):
    # Initialize variables
    loss = 0

    # Forward pass through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    # Create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    # Set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    # Determine if we are using teacher forcing this iteration
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # Forward batch of sequences through decoder one time step at a time
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # Teacher forcing: next input is current target
            decoder_input = target_variable[t].view(1, -1)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss

    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # No teacher forcing: next input is decoder's own current output
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss

    return loss, mask_loss, nTotal


def train(epoch, train_loader, encoder, decoder, encoder_optimizer, decoder_optimizer):
    # Ensure dropout layers are in train mode
    encoder.train()
    decoder.train()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    losses = AverageMeter()  # loss (per word decoded)

    start = time.time()

    print_losses = []
    n_totals = 0

    # Batches
    for i in range(train_loader.__len__()):
        input_variable, lengths, target_variable, mask, max_target_len = train_loader.__getitem__(i)
        # Zero gradients
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        # Set device options
        input_variable = input_variable.to(device)
        lengths = lengths.to(device)
        target_variable = target_variable.to(device)
        mask = mask.to(device)

        loss, mask_loss, nTotal = calc_loss(input_variable, lengths, target_variable, mask, max_target_len, encoder,
                                            decoder)
        print_losses.append(mask_loss.item() * nTotal)
        n_totals += nTotal

        # Perform backpropatation
        loss.backward()

        # Clip gradients: gradients are modified in place
        _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
        _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

        # Adjust model weights
        encoder_optimizer.step()
        decoder_optimizer.step()

        # Keep track of metrics
        losses.update(loss.item(), max_target_len)
        batch_time.update(time.time() - start)

        start = time.time()

        if i % print_every == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader),
                                                                  batch_time=batch_time,
                                                                  loss=losses))

    return sum(print_losses) / n_totals


def validate(val_loader, encoder, decoder):
    # Set dropout layers to eval mode
    encoder.eval()
    decoder.eval()

    # Initialize search module
    searcher = GreedySearchDecoder(encoder, decoder)

    # Batches
    for i, (input_array, target_array) in enumerate(val_loader[:10]):
        # Normalize sentence
        input_sentence = ' '.join([voc.index2word[idx.item()] for idx in input_array])
        print(input_sentence)

        # Evaluate sentence
        output_words = evaluate([input_array], searcher, voc, input_sentence)
        # Format and print response sentence
        output_words[:] = [x for x in output_words if not (x == '<end>' or x == '<pad>')]
        output_sentence = ''.join(output_words)
        print(output_sentence)


def main():
    word_map = json.load(open('data/WORDMAP.json', 'r'))
    n_words = len(word_map)

    train_loader = ChatbotDataset('train')
    val_loader = torch.utils.data.DataLoader(
        ChatbotDataset('valid'), batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    # Initialize word embeddings
    embedding = nn.Embedding(voc.num_words, hidden_size)

    # Initialize encoder & decoder models
    encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
    decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)

    # Use appropriate device
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Initialize optimizers
    print('Building optimizers ...')
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

    # Initializations
    print('Initializing ...')
    plot_losses = []

    # Epochs
    for epoch in range(start_epoch, epochs):
        # Run a training iteration with batch
        loss = train(epoch, train_loader, encoder, decoder, encoder_optimizer, decoder_optimizer)
        plot_losses.append(loss)

        validate(val_loader, encoder, decoder)


if __name__ == '__main__':
    main()
