from dataclasses import dataclass
import torch
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator


@dataclass
class TrainingArgs():
    num_epochs: int = 1
    learning_rate: float = 0.0002
    test_size: float = 0.9
    batch_size = 32
    model_name: str = "meta-llama/Llama-2-7b-chat-hf"
    lora_config: LoraConfig = None
    num_warmup_steps: int = 0
    mixed_precision: str = None
    accumulation_steps: int = 1


class Trainer():
    def __init__(
            self,
            model: torch.nn.Module,
            dataset,
            training_args: TrainingArgs
    ):

        # prepare the dataloaders
        self.dw = DatasetWrapper(
            model_name, dataset, training_args.batch_size, training_args.test_size)
        self.train_loader, self.val_loader = self.dw.prepare_data()

        # load model and apply lora
        self.model = model
        if training_args.lora_config is not None:
            self.model = get_peft_model(model, training_args.lora_config)

        # optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=training_args.learning_rate)

        # learning rate scheduler
        self.num_training_steps = len(self.train_loader)*self.epochs
        self.scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=self.num_training_steps,
        )

        # apply accelerator for mixed precision
        self.accelerator = Accelerator()
        self.train_loader,
        self.val_loader,
        self.model,
        self.optimizer,
        self.scheduler = self.accelerator.prepare(self.train_loader,
                                                  self.val_loader,
                                                  self.model,
                                                  self.optimizer,
                                                  self.scheduler)

    def train(self):

        self.model.train()

        progress_bar = tqdm(range(self.num_training_steps))

        for epoch in range(self.num_epochs):

            model.train()
            training_loss = 0
            for step, batch in enumerate(self.train_loader):
                outputs = model(**batch)
                loss = outputs.loss
                training_loss += loss.item()

                self.accelerator.backward(loss)

                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                if step % 100 == 0:
                    print(
                        f'Step {step}/{len(self.train_loader)} Training Loss: {training_loss/step*self.batch_size}')

                progress_bar.update(1)

            print(
                f'Epoch {epoch} Training Loss: {training_loss/len(self.train_loader)}')

            self.model.eval()
            val_loss = 0
            for step, batch in enumerate(self.val_loader):
                with torch.no_grad():
                    outputs = model(**batch)

                loss = outputs.loss
                validation_loss += loss.item()

                if step % 100 == 0:
                    print(
                        f'Step {step}/{len(self.val_loader)} Validation Loss: {val_loss/step*self.batch_size}')
            print(f'Epoch {epoch} Val Loss: {val_loss/len(self.val_loader)}')
