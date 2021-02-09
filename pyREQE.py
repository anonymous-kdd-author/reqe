import torch
import network, utils
import torch.nn as nn
import numpy as np
import six
from torch.autograd import Variable



class pyREQE(nn.Module):
    def __init__(self, num_nodes, args, hidden_dim, init_emb):

        super(pyREQE, self).__init__()

        # --
        # Define network
        self.num_node = num_nodes
        #self.graph = graph
        #self.degree_max = network.get_max_degree(self.graph)
        #self.degree = network.get_degree(self.graph)
        self.args = args
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(init_emb), freeze=False)
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(self.args.embedding_size, self.hidden_dim)
        self.lstm = nn.LSTM(input_size=self.args.embedding_size, hidden_size=self.hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, 1)

        # Define optimizer
        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=args.learning_rate)


    def forward(self, inputs):

        # node_ids: list of center node id; neighbor_ids: list of list of neighbor node ids;
        # seq_len: list of center node degree; labels: degree centrality score (used for guided loss)

        node_ids_sorted, neighbor_ids_sorted, seq_lens_sorted, _ = inputs


        # h0/c0: batch, 1, hidden_size
        #h0 = torch.zeros(1, node_ids_sorted.shape[0], self.hidden_dim)
        #c0 = torch.zeros(1, node_ids_sorted.shape[0], self.hidden_dim)

        #if self.args.cuda:
        #    h0 = h0.cuda()
        #    c0 = c0.cuda()
        self.h = self.init_hidden(node_ids_sorted.shape[0])  # initialize hidden state of GRU

        if self.args.cuda:
            self.h = self.h.cuda()
        lstm_input = self.embedding(neighbor_ids_sorted)
        lstm_input_pack = nn.utils.rnn.pack_padded_sequence(lstm_input, seq_lens_sorted, batch_first=True)
        gru_out, self.h = self.gru(lstm_input_pack, self.h)  # gru returns hidden state of all timesteps as well as hidden state at last timestep
        lstm_output = self.h[-1]

        m = nn.SELU()
        mlp_output = m(self.fc(lstm_output))
        return lstm_output, mlp_output.squeeze()




    def init_hidden(self, batch_size):
        if self.cuda:
            return Variable(torch.zeros((1, batch_size, self.hidden_dim))).cuda()
        else:
            return Variable(torch.zeros((1, batch_size, self.hidden_dim)))

    def train_step(self, inputs):
        self.optimizer.zero_grad()
        node_ids, neighbor_ids, seq_lens, neg_ids, labels = inputs
        node_ids = torch.LongTensor(node_ids)
        labels = torch.FloatTensor(labels)
        neighbor_ids = torch.LongTensor(neighbor_ids)
        seq_lens = torch.LongTensor(seq_lens)
        neg_ids = torch.LongTensor(neg_ids)
        sort_order = torch.argsort(seq_lens, descending=True)
        seq_lens_sort = seq_lens[sort_order]
        node_ids_sort = node_ids[sort_order]
        labels_sort = labels[sort_order]
        neighbor_ids_sort = neighbor_ids[sort_order, :]
        neg_ids_sort = neg_ids[sort_order, :]
        if self.args.cuda:
            node_ids_sort = node_ids_sort.cuda()
            neighbor_ids_sort = neighbor_ids_sort.cuda()
            seq_lens_sort = seq_lens_sort.cuda()
            labels_sort = labels_sort.cuda()
            neg_ids_sort = neg_ids_sort.cuda()
        inputs_new = [node_ids_sort, neighbor_ids_sort, seq_lens_sort, labels_sort]
        lstm_output, mlp_output = self(inputs_new)

        mse_lossfn = nn.MSELoss()
        l1_lossfn = nn.L1Loss()
        struc_loss = mse_lossfn(input=lstm_output, target=self.embedding(node_ids_sort))
        guided_loss = l1_lossfn(input=mlp_output, target=labels_sort)
        ns_loss = 0.0
        for i in range(neg_ids_sort.shape[1]):
            ns_loss += mse_lossfn(target=self.embedding(node_ids_sort), input=self.embedding(neg_ids_sort[:, i]))
        ns_loss *= self.args.alpha
        loss = struc_loss + self.args.lamb * guided_loss - self.args.alpha * ns_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.args.grad_clip)
        self.optimizer.step()
        # print(struc_loss.data, guided_loss.data, loss.data)
        return struc_loss.data, guided_loss.data, ns_loss.data

    def get_embedding(self):
        idx = torch.LongTensor(range(1, self.num_node))
        if self.args.cuda:
            idx = idx.cuda()
            return self.embedding(idx).cpu().detach().numpy()
        else:
            return self.embedding(idx).detach().numpy()

