import torch
import tqdm

from nn.betazero import BetaZero, BetaZeroConfig
import torch.optim as  optim


class NetWrapper:

    def __init__(self, net_config: BetaZeroConfig):
        self.board_size = net_config.board_size
        self.model = BetaZero(net_config)
        self.model.to(device=net_config.device)
        self.device = net_config.device
        self.optimizer = None

    def train(self, data, batch_size, n_iters, loss_visual_step=5, initiale_optimizer=True):

        self.model.train()

        if initiale_optimizer:
            self.optimizer = optim.Adam(self.model.parameters())

        total_loss = 0.
        actual_loss = 0.

        for i in range(1, n_iters + 1):

            board, policy, value = data.get_batch(batch_size)  # TODO get batch from dataset
            self.optimizer.zero_grad()

            v, p = self.model(torch.Tensor(board))
            loss = self.model.loss((v, p), (torch.Tensor(value), torch.Tensor(policy.reshape(-1,self.board_size*self.board_size))))
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            actual_loss += loss.item()

            if i % loss_visual_step == 0:
                print(f"{i} loss: {(actual_loss / loss_visual_step):.3f}")
                actual_loss = 0

        return total_loss / n_iters

    def initial_train(self, data, n_iters=1, loss_visual_step=10):

        self.model.train()
        self.optimizer = optim.Adam(self.model.parameters())

        total_loss = 0.
        actual_loss = 0.

        for i in range(1, n_iters + 1):
            iter_loss = 0.
            for j in tqdm.tqdm(range(len(data))):

                board, policy, value = data.get_batch(1)  # TODO get batch from dataset
                self.optimizer.zero_grad()

                v, p = self.model(torch.Tensor(board))
                loss = self.model.loss((v, p), (
                torch.Tensor(value), torch.Tensor(policy.reshape(-1, self.board_size * self.board_size))))
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                actual_loss += loss.item()
                iter_loss += loss.item()

                if i % loss_visual_step == 0:
                    print(f"\t{i} loss: {(actual_loss / loss_visual_step):.3f}")
                    actual_loss = 0

            print(f"{i} iteration loss: {iter_loss/len(data)}")

        return total_loss / n_iters



    def predict(self, board):
        self.model.eval()
        with torch.no_grad():
            v, p = self.model(torch.Tensor(board))

        value = v.detatch().numpy()
        probs = p.detatch().numpy().reshape(self.board_size, self.board_size)

        return value, probs

    def save_model(self, folder="C:\\Users\\user\\Documents\\GomokuProject\\resources\\models", name="model"):

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'oprimizer_state_dict': self.optimizer.state_dict()
        }, f"{folder}/{name}")

    def load_model(self, path, load_optim=False):
        cp = torch.load(path)
        self.model.load_state_dict(cp['model_state_dict'])
        if load_optim:
            self.optimizer = optim.Adam(self.model.parameters())
            self.optimizer.load_state_dict(cp['optimizer_state_dict'])
        return self.model
