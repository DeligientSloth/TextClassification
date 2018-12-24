class ResnetBlock(nn.Module):
    def __init__(self, channel_size):
        super(ResnetBlock, self).__init__()

        self.channel_size = channel_size
        self.maxpool = nn.Sequential(
            nn.ConstantPad1d(padding=(0, 1), value=0),
            nn.MaxPool1d(kernel_size=3, stride=2)
        )
        self.conv = nn.Sequential(
            nn.BatchNorm1d(num_features=self.channel_size),
            nn.ReLU(),
            nn.Conv1d(self.channel_size, self.channel_size, kernel_size=3, padding=1),

            nn.BatchNorm1d(num_features=self.channel_size),
            nn.ReLU(),
            nn.Conv1d(self.channel_size, self.channel_size, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x_shortcut = self.maxpool(x)
        x = self.conv(x_shortcut)
        x = x + x_shortcut
        #         print('resnet:{}'.format(x.shape))
        return x


class NeuralNet(nn.Module):
    """
    DPCNN model, 3
    1. region embedding: using TetxCNN to generte
    2. two 3 conv(padding) block
    3. maxpool->3 conv->3 conv with resnet block(padding) feature map: len/2
    """

    def __init__(self):
        super(NeuralNet, self).__init__()
        self.channel_size = 250

        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False

        # region embedding
        self.region_embedding = nn.Sequential(
            nn.Conv1d(embed_size, self.channel_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=self.channel_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.conv_block = nn.Sequential(
            nn.BatchNorm1d(num_features=self.channel_size),
            nn.ReLU(),
            nn.Conv1d(self.channel_size, self.channel_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=self.channel_size),
            nn.ReLU(),
            nn.Conv1d(self.channel_size, self.channel_size, kernel_size=3, padding=1),
        )

        self.seq_len = maxlen
        resnet_block_list = []
        while (self.seq_len > 2):
            resnet_block_list.append(ResnetBlock(self.channel_size))
            self.seq_len = self.seq_len // 2
        #         print('seqlen{}'.format(self.seq_len))
        self.resnet_layer = nn.Sequential(*resnet_block_list)

        self.linear_out = nn.Linear(self.seq_len * self.channel_size, 1)  # 改成输出一个值

    def forward(self, x):
        batch = x.shape[0]
        print("===========start==============")

        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = self.region_embedding(x)
        #         print(x.shape)

        x = self.conv_block(x)
        #         print(x.shape)

        x = self.resnet_layer(x)

        x = x.permute(0, 2, 1)
        #         print(x.shape)
        #         print("===========end==============")
        x = x.contiguous().view(x.size(0), -1)
        out = self.linear_out(x)
        return out
