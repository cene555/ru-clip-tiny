import torch
from transformers import get_linear_schedule_with_warmup
from tqdm.auto import tqdm
from .dataset import RuCLIPTinyDataset


class Trainer:
    def __init__(self, train_dataframe, train_dir,
                 val_dataframe=None, val_dir=None, learning_rate=1e-4,
                 freeze_image_encoder=True, freeze_text_encoder=False, max_text_len=77,
                 train_batch_size=64, val_batch_size=64, num_workers=2,
                 weight_decay=1e-4, grad_accum=8):
        self.train_dataframe = train_dataframe
        self.train_dir = train_dir
        self.val_dataframe = val_dataframe
        self.val_dir = val_dir
        self.learning_rate = learning_rate
        self.freeze_image_encoder = freeze_image_encoder
        self.freeze_text_encoder = freeze_text_encoder
        self.max_text_len = max_text_len
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.weight_decay = weight_decay
        self.grad_accum = grad_accum
        print(f"train batch size = {self.train_batch_size * self.grad_accum}")

        def train_model(self, model, epochs_num=1, device='cuda', verbose=10):

            is_val = self.val_dataframe is not None and self.val_dir is not None

            model.to(device)

            train_dataset = RuCLIPTinyDataset(self.train_dir, self.train_dataframe, self.max_text_len)

            train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                       batch_size=self.train_batch_size,
                                                       shuffle=True,
                                                       pin_memory=True,
                                                       num_workers=self.num_workers)

            if is_val:
                val_dataset = RuCLIPTinyDataset(self.val_dir, self.val_dataframe, self.max_text_len)
                val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                         batch_size=self.val_batch_size,
                                                         shuffle=False,
                                                         pin_memory=True,
                                                         num_workers=self.num_workers)

            for i, child in enumerate(model.children()):
                if (i == 0 and self.freeze_image_encoder) or (i == 1 and self.freeze_text_encoder):
                    for param in child.parameters():
                        param.requires_grad = False

            loss_img = torch.nn.CrossEntropyLoss()
            loss_txt = torch.nn.CrossEntropyLoss()

            optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate, betas=(0.9, 0.98), eps=1e-8,
                                          weight_decay=self.weight_decay)
            total_steps = len(train_loader) * epochs_num
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                                        num_warmup_steps=0,
                                                        num_training_steps=total_steps)

            for epoch in range(epochs_num):
                model.train()
                print(f'start training epoch {epoch}')
                curr_batch = 0
                X = []
                Y = []
                curr_batch = 0
                for i, batch in enumerate(tqdm(train_loader)):
                    images = batch[0].cuda()
                    input_ids = batch[1].cuda()
                    attention_mask = batch[2].cuda()

                    image_features = model.encode_image(images)
                    text_features = model.encode_text(input_ids, attention_mask)

                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                    X.append(image_features)
                    Y.append(text_features)

                    if ((i + 1) % accum_iter == 0) or (i + 1 == len(train_loader)):
                        optimizer.zero_grad()
                        X = torch.cat(X, axis=0).cuda()
                        Y = torch.cat(Y, axis=0).cuda()
                        logit_scale = clip_model.logit_scale.exp()
                        logits_per_image = logit_scale * X @ Y.t()
                        logits_per_text = logits_per_image.t()
                        ground_truth = torch.arange(X.shape[0], dtype=torch.long).cuda()
                        img_l = loss_img(logits_per_image, ground_truth)
                        text_l = loss_txt(logits_per_text, ground_truth)
                        total_loss = (img_l + text_l) / 2
                        if curr_batch % verbose == 0:
                            print(f'{i}/{len(train_loader)} total_loss {total_loss}')
                        total_loss.backward()
                        optimizer.step()
                        scheduler.step()
                        torch.nn.utils.clip_grad_norm_(clip_model.parameters(), 2)
                        X = []
                        Y = []
                if is_val:
                    print(f'start val epoch {epoch}')
                    total_loss = 0
                    model.eval()
                    with torch.no_grad():
                        for i, batch in enumerate(tqdm(val_loader)):
                            images = batch[0].to(device)
                            input_ids = batch[1].to(device)
                            attention_mask = batch[2].to(device)

                            logits_per_image, logits_per_text = model(images, input_ids, attention_mask)
                            ground_truth = torch.arange(batch[1].shape[0], dtype=torch.long).to(device)
                            img_l = loss_img(logits_per_image, ground_truth).item()
                            text_l = loss_txt(logits_per_text, ground_truth).item()
                            total_loss += (img_l + text_l) / 2
                        print(f'val loss = {total_loss / len(val_loader)}')
            return model
