@dataclass
class TrainingArgs():
    num_epochs: int = 1
    learning_rate: float = 0.0002
    test_size: float = 0.9
    batch_size= 32
    model_name : str = "meta-llama/Llama-2-7b-chat-hf"
    lora_config: LoraConfig = None
    num_warmup_steps: int = 0
    mixed_precision: str = None
    accumulation_steps: int = 1


class Trainer():
    def __init__(
            self,
            model: torch.nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
            training_args: TrainingArgs
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader

        #load model and apply lora
        self.model = model
        if training_args.lora_config is not None:
            self.model = get_peft_model(model, training_args.lora_config)
            self.model = prepare_model_for_kbit_training(self.model)

        #optimizer
        self.optimizer = bnb.optim.Adam8bit(model.parameters(), lr = training_args.learning_rate)

        #learning rate scheduler
        self.num_epochs = training_args.num_epochs
        self.num_training_steps = len(train_loader)*self.num_epochs
        self.scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=self.num_training_steps,
        )


        #apply accelerator for mixed precision
        self.accelerator=Accelerator()
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
                training_loss+=loss.item()
                self.accelerator.backward(loss)

                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                if step % 100 == 0:
                    print(f'Step {step}/{len(self.train_loader)} Training Loss: {training_loss/step*self.batch_size}')

                progress_bar.update(1)

            print(f'Epoch {epoch} Training Loss: {training_loss/len(self.train_loader)}')

            self.model.eval()
            val_loss = 0
            for step, batch in enumerate(self.val_loader):
                with torch.no_grad():
                    outputs=model(**batch)

                loss = outputs.loss
                validation_loss+=loss.item()

                if step % 100 == 0:
                    print(f'Step {step}/{len(self.val_loader)} Validation Loss: {val_loss/step*self.batch_size}')
            print(f'Epoch {epoch} Val Loss: {val_loss/len(self.val_loader)}')

