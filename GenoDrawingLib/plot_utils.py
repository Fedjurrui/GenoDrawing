
from PIL import Image, ImageDraw
import torch
import torchvision



def store_18_examples_AE(test_set,encoder,decoder,device):
    print(len(test_set))
    image_examples = [i[0] for i in test_set]
    image_examples = torch.stack(image_examples).to(device)
    encoder.eval()
    decoder.eval()
    encoded_data = encoder(image_examples)
    image_examples.detach()
   # Decode data
    decoded_data = image_examples
    decoded_predictions = decoder(encoded_data)

    new = Image.new("RGB",(1800,900))
    x = 0
    y = 0
    for i in range(9):
        img = torchvision.transforms.ToPILImage()(decoded_data[i])
        draw = ImageDraw.Draw(img)
        new.paste(img,(x,y))
        pred = torchvision.transforms.ToPILImage()(decoded_predictions[i])
        pred_draw = ImageDraw.Draw(pred)
        new.paste(pred, (x + 300, y))
        if x < 1800:
            x+= 600
        if (x >= 1800) & (y < 900):
            x = 0
            y+= 300
    return new

