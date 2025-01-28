
from torch import nn
from torch.nn import functional as F
from .architecture_factory import register_architecture

@register_architecture('mlp2hidden')
class MLP2Hidden(nn.Module):

    def __init__(self, **kwargs):
        super(MLP2Hidden, self).__init__()
        model_params = kwargs['model_params']
        input_dim = model_params['input']
        hidden1_dim = model_params['hidden1']
        hidden2_dim = model_params['hidden2']
        dropout = model_params['dropout']
        output_dim = model_params['output']

        self.fc1 = nn.Linear(input_dim, hidden1_dim)
        self.fc2 = nn.Linear(hidden1_dim, hidden2_dim)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden2_dim, output_dim)

    def forward(self, batch):
        x = F.relu(self.fc1(batch))
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = self.out(x)
        return x

    def freeze(self):
        self.fc1.requires_grad_(False)

    def freeze_all(self):
        self.fc1.requires_grad_(False)
        self.out.requires_grad_(False)

    def unfreeze_all(self):
        self.fc1.requires_grad_(True)
        self.out.requires_grad_(True)

@register_architecture('mlp3hidden')
class MLP3Hidden(nn.Module):

    def __init__(self, **kwargs):
        super(MLP3Hidden, self).__init__()
        model_params = kwargs['model_params']
        input_dim = model_params['input']
        hidden1_dim = model_params['hidden1']
        hidden2_dim = model_params['hidden2']
        hidden3_dim = model_params['hidden3']
        dropout = model_params['dropout']
        output_dim = model_params['output']

        self.fc1 = nn.Linear(input_dim, hidden1_dim)
        self.fc2 = nn.Linear(hidden1_dim, hidden2_dim)
        self.fc3 = nn.Linear(hidden2_dim, hidden3_dim)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden3_dim, output_dim)

    def forward(self, batch):
        x = F.relu(self.fc1(batch))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.drop(x)
        x = self.out(x)
        return x

    def freeze(self):
        self.fc1.requires_grad_(False)

    def freeze_all(self):
        self.fc1.requires_grad_(False)
        self.out.requires_grad_(False)

    def unfreeze_all(self):
        self.fc1.requires_grad_(True)
        self.out.requires_grad_(True)


@register_architecture('mlp4hidden')
class MLP4Hidden(nn.Module):

    def __init__(self, **kwargs):
        super(MLP4Hidden, self).__init__()
        model_params = kwargs['model_params']
        input_dim = model_params['input']
        hidden1_dim = model_params['hidden1']
        hidden2_dim = model_params['hidden2']
        hidden3_dim = model_params['hidden3']
        hidden4_dim = model_params['hidden4']
        dropout = model_params['dropout']
        output_dim = model_params['output']

        self.fc1 = nn.Linear(input_dim, hidden1_dim)
        self.fc2 = nn.Linear(hidden1_dim, hidden2_dim)
        self.fc3 = nn.Linear(hidden2_dim, hidden3_dim)
        self.fc4 = nn.Linear(hidden3_dim, hidden4_dim)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden4_dim, output_dim)

    def forward(self, batch):
        x = F.relu(self.fc1(batch))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.drop(x)
        x = self.out(x)
        return x

    def freeze(self):
        self.fc1.requires_grad_(False)

    def freeze_all(self):
        self.fc1.requires_grad_(False)
        self.out.requires_grad_(False)

    def unfreeze_all(self):
        self.fc1.requires_grad_(True)
        self.out.requires_grad_(True)