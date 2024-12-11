import torch
import torch.nn as nn
torch.autograd.set_detect_anomaly(True)


import eq
#import eq.distributions as dist

from .tpp_model import TPPModel


class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden=32, num_hidden=2, sigma=0.01, activation=nn.Tanh()):
        super().__init__()
        if num_hidden == 0:
            self.linears = nn.ModuleList([nn.Linear(dim_in, dim_out)])
        else:
            self.linears = nn.ModuleList()
            self.linears.append(nn.Linear(dim_in, dim_hidden))
            self.linears.extend([nn.Linear(dim_hidden, dim_hidden) for _ in range(num_hidden - 1)])
            self.linears.append(nn.Linear(dim_hidden, dim_out))
        
        for m in self.linears:
            nn.init.normal_(m.weight, mean=0, std=sigma)
            nn.init.uniform_(m.bias, a=-sigma, b=sigma)
        
        self.activation = activation

    def forward(self, x):
        for layer in self.linears[:-1]:
            x = self.activation(layer(x))
        return self.linears[-1](x)


class NJDTPP(TPPModel):
    def __init__(
        self,
        dim_eta: int,
        dim_hidden: int = 32,
        num_hidden: int = 2,
        sigma: float = 0.01,
        activation: nn.Module = nn.Tanh(),
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.dim_eta = dim_eta
        self.dim_hidden = dim_hidden
        self.num_hidden = num_hidden
        self.sigma = sigma
        self.activation = activation
        self.learning_rate = learning_rate

        # Drift, Diffusion, and Jump Nets
        self.F = MLP(dim_eta, dim_eta, dim_hidden, num_hidden, sigma, activation)
        self.G = MLP(dim_eta, dim_eta, dim_hidden, num_hidden, sigma, activation)
        self.H = MLP(dim_eta, dim_eta * dim_eta, dim_hidden, num_hidden, sigma, activation)

        self.eta0 = nn.Parameter(torch.empty(dim_eta).normal_(mean=0, std=0.1)).requires_grad_()


    def drift(self, y):
        return self.F(y) #.view(-1, self.dim_eta) ou .view(self.batch_size, self.dim_eta)

    def diffusion(self, y):
        return self.G(y) #.view(-1, self.dim_eta) ou .view(self.batch_size, self.dim_eta)

    def jump(self, y):
        return self.H(y).view(y.size(0), self.dim_eta, self.dim_eta) #.view(-1, self.dim_eta, self.dim_eta) ou .view(self.batch_size, self.dim_eta, self.dim_eta)

        
    def euler_solver(self, eta_initial, adjacent_events, num_divide, device):
        """Perform Euler integration for a single time interval."""
        #added line
        #adjacent_events = torch.tensor(adjacent_events, device=device, dtype=torch.float32)

        dt = torch.diff(adjacent_events, dim=1) / num_divide
        ts = torch.cat([adjacent_events[:, 0].unsqueeze(dim=1) + dt * j for j in range(num_divide + 1)], dim=1)

        #eta_ts = eta_initial.unsqueeze(2)
        eta_ts = torch.Tensor().to(device)
        eta_ts = torch.cat((eta_ts, eta_initial.unsqueeze(2)), dim=2)
        for _ in range(num_divide):
            eta_output = eta_initial + self.drift(eta_initial.clone())*dt + self.diffusion(eta_initial.clone())*torch.sqrt(dt)*torch.randn_like(eta_initial).to(device)
            eta_ts = torch.cat((eta_ts, eta_output.unsqueeze(2)), dim=2)
            eta_initial = torch.clamp(eta_output, min=-10, max=10) #eta_output  # clipped here because of huge numbers in integral and sum of the loss
                                                                                # leading to nan in the training loss
        return ts, eta_ts

    def forward(self, time_seqs, type_seqs, mask, batch_size, num_divide=10):
        device = self.eta0.device
        padded_seq_length = time_seqs.size(1)

        eta_batch_l = torch.zeros(batch_size, padded_seq_length, self.dim_eta, device=device)
        eta_batch_r = torch.zeros(batch_size, padded_seq_length, self.dim_eta, device=device)
        eta_time_l = torch.zeros(batch_size, padded_seq_length, device=device)
        eta_time_r = torch.zeros(batch_size, padded_seq_length, device=device)

        eta_batch_l[:, 0, :] = self.eta0.unsqueeze(0).repeat(batch_size, 1)
        event_type = type_seqs[:, 0].tolist()
        eta_time_l[:, 0] = eta_batch_l[list(range(0, batch_size)), 0, event_type] # torch.arange(batch_size)

        eta_batch_r[:, 0, :] = eta_batch_l[:, 0, :] + self.jump(eta_batch_l[:, 0, :].clone())[list(range(0, batch_size)), :, event_type] # torch.arange(batch_size)
        eta_time_r[:, 0] = eta_batch_r[list(range(0, batch_size)), 0, event_type] # torch.arange(batch_size)
        
        tsave = torch.Tensor().to(device)
        eta_tsave = torch.Tensor().to(device)
        eta_initial = eta_batch_r[:, 0, :]

        for i in range(padded_seq_length - 1):
            adjacent_events = time_seqs[:, i:i + 2]
            ts, eta_ts_l = self.euler_solver(eta_initial, adjacent_events, num_divide, device)
            tsave = torch.cat((tsave, ts), dim=1) 
            eta_tsave = torch.cat((eta_tsave, eta_ts_l), dim=2)
                       
            eta_batch_l[:, i+1, :] = eta_ts_l[:, :, -1]

            eta_ts_r = eta_ts_l.clone()
            event_type = type_seqs[:, i+1].tolist()
            eta_ts_r[:, :, -1] = eta_ts_l[:, :, -1] + self.jump(eta_ts_l[:, :, -1])[list(range(0, batch_size)), :, event_type] # torch.arange(batch_size)

            eta_batch_r[:, i+1, :] = eta_ts_r[:, :, -1]

            eta_time_l[:, i+1] = eta_batch_l[list(range(0, batch_size)), i+1, event_type] #list(range(0, batch_size)) à la place des torch
            eta_time_r[:, i+1] = eta_batch_r[list(range(0, batch_size)), i+1, event_type] # torch.arange(batch_size)

            eta_initial = eta_ts_r[:, :, -1]

        masked_eta_time_l = eta_time_l * mask
        sum_term = torch.sum(masked_eta_time_l)

        mask_without_first_col = mask[:, 1:]
        expanded_mask = mask_without_first_col.unsqueeze(2).repeat(1, 1, num_divide+1).view(mask.size(0), -1)
        expanded_mask = expanded_mask.unsqueeze(1).repeat(1,self.dim_eta,1)

        eta_tsave = eta_tsave * expanded_mask # mask the eta_tsave

        expanded_diff_tsave = torch.diff(tsave).unsqueeze(1).repeat(1, self.dim_eta, 1)
        integral_term = torch.sum(
            (torch.exp(eta_tsave)[:, :, :-1] * expanded_mask[:, :, :-1] \
             + torch.exp(eta_tsave)[:, :, 1:] * expanded_mask[:, :, 1:]) * (expanded_diff_tsave * expanded_mask[:, :, 1:])
        ) / 2  # reason for mask: e^0=1

        log_likelihood = sum_term - integral_term
        return -log_likelihood, sum_term, integral_term



    def nll_loss(self, batch: eq.data.Batch):
        # Compute the loss and its components
        loss, sum_term, integral_term = self.forward(batch.arrival_times, batch.mag_type, batch.mask, batch.batch_size)
        print(f"Loss = {loss.item()}, Sum = {sum_term.item()}, Integral = {integral_term.item()}")
        return loss

    
    def validation_step(self, batch, batch_idx):
        loss = self.nll_loss(batch) #.mean()
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        base_params = [p for p in self.parameters() if p is not self.eta0]
        return torch.optim.Adam([
            {'params': base_params},
            {'params': [self.eta0], 'lr': 1.0e-1, 'weight_decay': 1e-5},
        ], lr=1e-3, weight_decay=1e-5)
    








    #***** FUNCTION TO DEBUG after we solved the training problem *****************

    def sample(self, batch_size: int, duration: float, t_start: float = 0.0, **kwargs):
        """Generate a batch of event sequences from the model."""
        device = self.eta0.device
        t_end = t_start + duration

        eta = self.eta0.unsqueeze(0).expand(batch_size, -1).to(device)
        arrival_times = torch.zeros(batch_size, 0, device=device)
        inter_times = torch.zeros(batch_size, 0, device=device)

        while True:
            drift = self.drift(eta)
            diffusion = self.diffusion(eta)
            dt = drift + diffusion * torch.randn_like(drift)
            next_inter_times = torch.relu(dt)
            next_arrival_times = (arrival_times[:, -1:] if arrival_times.size(1) > 0 else t_start) + next_inter_times

            if (next_arrival_times >= t_end).all():
                break

            arrival_times = torch.cat([arrival_times, next_arrival_times], dim=1)
            inter_times = torch.cat([inter_times, next_inter_times], dim=1)

        mask = (arrival_times < t_end).float()
        arrival_times = arrival_times * mask
        inter_times = inter_times * mask

        batch = eq.data.Batch(
            inter_times=inter_times,
            arrival_times=arrival_times,
            t_start=torch.full([batch_size], t_start, device=device),
            t_end=torch.full([batch_size], t_end, device=device),
            mask=mask,
        )
        return batch

    def evaluate_intensity(self, sequence: eq.data.Sequence, num_grid_points: int = 100):
        """Evaluate the intensity function."""
        device = self.eta0.device
        times = torch.linspace(sequence.t_start, sequence.t_end, num_grid_points, device=device)
        intensities = torch.zeros_like(times)

        eta = self.eta0
        for i, t in enumerate(times):
            drift = self.drift(eta)
            diffusion = self.diffusion(eta)
            eta = eta + drift * (t - times[i - 1]) + diffusion * torch.sqrt(t - times[i - 1])
            intensities[i] = torch.norm(eta)

        return times, intensities

    def evaluate_compensator(self, sequence: eq.data.Sequence, num_grid_points: int = 50):
        """Evaluate the compensator function."""
        device = self.eta0.device
        times = torch.linspace(sequence.t_start, sequence.t_end, num_grid_points, device=device)
        compensators = torch.zeros_like(times)

        eta = self.eta0
        for i, t in enumerate(times):
            drift = self.drift(eta)
            diffusion = self.diffusion(eta)
            eta = eta + drift * (t - times[i - 1]) + diffusion * torch.sqrt(t - times[i - 1])
            compensators[i] = torch.sum(eta ** 2)  # Example: Sum of squared eta as compensator

        return times, compensators



    ''' # my old forward
    def forward(self, time_seqs, type_seqs, mask, batch_size, num_divide=10):
        """Long forward pass for loss calculation using Euler solver."""
        padded_seq_length = time_seqs.size(1)
        device = self.eta0.device #time_seqs.device

        eta_batch_l = torch.zeros(batch_size, padded_seq_length, self.dim_eta, device=device)
        eta_batch_r = torch.zeros(batch_size, padded_seq_length, self.dim_eta, device=device)
        eta_time_l = torch.zeros(batch_size, padded_seq_length, device=device)
        eta_time_r = torch.zeros(batch_size, padded_seq_length, device=device)

        eta_batch_l[:, 0, :] = self.eta0.unsqueeze(0).repeat(batch_size, 1)
        event_type = type_seqs[:, 0]
        eta_time_l[:, 0] = eta_batch_l[torch.arange(batch_size), 0, event_type]

        eta_batch_r[:, 0, :] = eta_batch_l[:, 0, :].clone() + self.jump(eta_batch_l[:, 0, :].clone())[torch.arange(batch_size), :, event_type]
        eta_time_r[:, 0] = eta_batch_r[torch.arange(batch_size), 0, event_type]

        for i in range(padded_seq_length - 1):
            adjacent_events = time_seqs[:, i:i + 2]
            dt = torch.diff(adjacent_events, dim=1) / num_divide
            eta_initial = eta_batch_r[:, i, :].clone().to(device)

            for _ in range(num_divide):
                eta_initial = eta_initial.clone().to(device) + self.drift(eta_initial.clone().to(device)) * dt + self.diffusion(eta_initial.clone().to(device)) * torch.sqrt(dt) * torch.randn_like(eta_initial.clone().to(device))
                #eta_initial = torch.clamp(eta_initial, min=-10, max=10)  # Add clamping

            eta_batch_l[:, i + 1, :] = eta_initial.clone().to(device)
            event_type = type_seqs[:, i + 1].clone().to(device)
            eta_batch_r[:, i + 1, :] = eta_initial.clone() + self.jump(eta_initial.clone())[torch.arange(batch_size), :, event_type]

            eta_time_l[:, i + 1] = eta_batch_l[torch.arange(batch_size), i + 1, event_type]
            eta_time_r[:, i + 1] = eta_batch_r[torch.arange(batch_size), i + 1, event_type]

        #assert mask.shape == eta_time_l.shape, "Mask dimensions do not align with eta_time_l"
        masked_eta_time_l = eta_time_l * mask
        sum_term = torch.sum(masked_eta_time_l)
        #correction for dim, is it legitimate ? is it correct ?
        eta_time_r_reduced = eta_time_r[:, :-1].clone()
        integral_term = torch.sum(eta_time_r_reduced * torch.diff(time_seqs, dim=1) * mask[:, 1:])
        log_likelihood = sum_term - integral_term
        #return -log_likelihood
        return -log_likelihood, sum_term, integral_term
    '''