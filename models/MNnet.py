import torch.nn as nn

class Net4(nn.Module):            # the best
    def __init__(self):
        super(Net4, self).__init__()
        
        self.cnn_layers = nn.Sequential(
            
            # Layer 1
            nn.Conv2d(1, 128, kernel_size=(5, 3), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            # nn.MaxPool2d((2, 2), stride = 2),

            # Layer 2
            nn.Conv2d(128, 128, kernel_size=(5, 3), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d((2, 2), stride = 2),

            # Layer 3
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d((2, 2), stride = 2),

            # Layer 4
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d((2, 2), stride = 2),

            # Layer 5
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d((2, 2), stride = 1),

        )

        self.linear_layers = nn.Sequential(
            nn.Linear(2688, 26),
            nn.Linear(26, 1)
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        # print(x.size())
        x = self.linear_layers(x)
        return x



class Net5(nn.Module):            # the best
    def __init__(self):
        super(Net5, self).__init__()
        
        self.cnn_layers = nn.Sequential(
            
            # Layer 1
            nn.Conv2d(1, 4, kernel_size=(5, 3), stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.1, inplace=True),
            # nn.MaxPool2d((2, 2), stride = 2),

            # Layer 2
            nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d((2, 2), stride = 2),

            # Layer 3
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d((2, 2), stride = 2),

            # Layer 4
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d((2, 2), stride = 2),

            # Layer 5
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d((2, 2), stride = 1),

        )

        self.linear_layers = nn.Sequential(
            nn.Linear(1344, 26),
            nn.Linear(26, 1)
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        # print(x.size())
        x = self.linear_layers(x)
        return x
        
        
 
class Net6(nn.Module):            # the best
    def __init__(self):
        super(Net6, self).__init__()
        
        self.cnn_layers = nn.Sequential(
            
            # Layer 1
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=3),

            # Layer 2
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=3),

            # Layer 3
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=3),

        )

        self.linear_layers = nn.Sequential(
            nn.Linear(256, 100),
            nn.Linear(100, 1)
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        # print(x.size())
        x = self.linear_layers(x)
        return x



class Net3(nn.Module):            # the best
    def __init__(self):
        super(Net3, self).__init__()
        
        self.cnn_layers = nn.Sequential(
            
            # Layer 1
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.1, inplace=True),
            # nn.MaxPool2d(kernel_size=2),

            # Layer 2
            nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.1, inplace=True),
            # nn.MaxPool2d(kernel_size=2),

            # Layer 3
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True),
            # nn.MaxPool2d(kernel_size=2),

            # Layer 4
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            # nn.MaxPool2d(kernel_size=2),

            # Layer 5
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            # nn.MaxPool2d(kernel_size=2),

            # Layer 6
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2),


            # Layer 7
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2),


            # Layer 8
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2),


            # Layer 9
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2),

        )

        self.linear_layers = nn.Sequential(
            nn.Linear(1792, 150),
            nn.Linear(150, 1)
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        # print(x.size())
        x = self.linear_layers(x)
        return x
               
        

class Net1(nn.Module):            # the best
    def __init__(self):
        super(Net1, self).__init__()
        
        self.real_fconv = nn.Conv2d(1, 126, kernel_size = (1, 32), padding = 0)
        self.imag_fconv = nn.Conv2d(1, 126, kernel_size = (1, 32), padding = 0)

        real_W = initialize_real(126, 32)
        imag_W = initialize_imag(126, 32)

        self.real_fconv.weight = torch.nn.Parameter(real_W)
        self.imag_fconv.weight = torch.nn.Parameter(imag_W)

        self.batch_norm = nn.BatchNorm2d(126)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

        self.cnn_layers = nn.Sequential(

            # Layer 1
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.1, inplace=True),
            # nn.MaxPool2d(kernel_size=2),

            # Layer 2
            nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.1, inplace=True),
            # nn.MaxPool2d(kernel_size=2),

            # Layer 3
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True),
            # nn.MaxPool2d(kernel_size=2),

            # Layer 4
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            # nn.MaxPool2d(kernel_size=2),

            # Layer 5
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # Layer 6
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2),


            # Layer 7
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2),


            # Layer 8
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2),


            # Layer 9
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2),

        )


        self.linear_layers = nn.Sequential(
            nn.Linear(1152, 100),
            nn.Linear(100, 1)
        )

    # Defining the forward pass    
    def forward(self, x):
        real = self.relu(self.batch_norm(self.real_fconv(x)))
        imag = self.relu(self.batch_norm(self.imag_fconv(x)))
        x = real + imag
        x = x.transpose_(1, 3)
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        # print(x.size())
        x = self.linear_layers(x)
        return x
        



class Net0(nn.Module):            # the best
    def __init__(self):
        super(Net0, self).__init__()
        
        self.cnn_layers = nn.Sequential(
            
            # Layer 1
            nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(2),
            nn.LeakyReLU(0.1, inplace=True),
            # nn.MaxPool2d(kernel_size=2),

            # Layer 2
            nn.Conv2d(2, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.1, inplace=True),
            # nn.MaxPool2d(kernel_size=2),

            # Layer 3
            nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.1, inplace=True),
            # nn.MaxPool2d(kernel_size=2),

            # Layer 4
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True),
            # nn.MaxPool2d(kernel_size=2),

            # Layer 5
            nn.Conv2d(16, 24, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(24),
            nn.LeakyReLU(0.1, inplace=True),
            # nn.MaxPool2d(kernel_size=2),

            # Layer 6
            nn.Conv2d(24, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            # nn.MaxPool2d(kernel_size=2),

            # Layer 7
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            # nn.MaxPool2d(kernel_size=2),

            # Layer 8
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2),


            # Layer 9
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2),


            # Layer 10
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2),


            # Layer 11
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2),

        )

        self.linear_layers = nn.Sequential(
            nn.Linear(1792, 150),
            nn.Linear(150, 1)
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        # print(x.size())
        x = self.linear_layers(x)
        return x
        
        


class MNNET(nn.Module):                                       # working with two initialized kernels that function as FTFS and then using preprocessing step to return into real
    def __init__(self):
        super(MNNET, self).__init__()
      
        self.real_fconv = nn.Conv2d(1, 126, kernel_size = (1, 32), padding = 0)
        self.imag_fconv = nn.Conv2d(1, 126, kernel_size = (1, 32), padding = 0)

        real_W = initialize_real(126, 32)
        imag_W = initialize_imag(126, 32)

        self.real_fconv.weight = torch.nn.Parameter(real_W)
        self.imag_fconv.weight = torch.nn.Parameter(imag_W)

        self.batch_norm = nn.BatchNorm2d(126)
        self.relu = nn.LeakyReLU(0.1)

        self.final_convs = nn.Sequential(

            # Layer 1
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.1, inplace=True),
            # nn.MaxPool2d(kernel_size=2),

            # Layer 2
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.1, inplace=True),
            # nn.MaxPool2d(kernel_size=2),

            # Layer 2
            nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.1, inplace=True),
            # nn.MaxPool2d(kernel_size=2),

            # Layer 3
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True),
            # nn.MaxPool2d(kernel_size=2),

            # Layer 4
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # Layer 5
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # Layer 6
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2),


            # Layer 7
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2),


            # Layer 8
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2),


            # Layer 9
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2),

        )
        
        
        self.linear_layers = nn.Sequential(
            nn.Linear(2048, 100),
            nn.Linear(100, 1)
        )
        

    # Defining the forward pass    
    def forward(self, x):
        real = self.real_fconv(x)
        imag = self.imag_fconv(x)
        real = torch.pow(real, 2)
        imag = torch.pow(imag, 2)
        x = torch.cat((real, imag), dim=3)
        x = torch.sum(x, 3)
        x = x.unsqueeze(1)
        x = torch.sqrt(x)
        x = self.relu(x)
        x = self.final_convs(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x.reshape(-1, 1)




class Net3_D(nn.Module):            # the best
    def __init__(self):
        super(Net3_D, self).__init__()
        
        self.dilated_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=2, dilation = 2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1)
        )

        self.cnn_layers = nn.Sequential(
            
            # Layer 1
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True),
            # nn.MaxPool2d(kernel_size=2),

            # Layer 2
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True),
            # nn.MaxPool2d(kernel_size=2),

            # Layer 3
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True),
            # nn.MaxPool2d(kernel_size=2),

            # Layer 4
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            # nn.MaxPool2d(kernel_size=2),

            # Layer 5
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            # nn.MaxPool2d(kernel_size=2),

            # Layer 6
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2),


            # Layer 7
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2),


            # Layer 8
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2),


            # Layer 9
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2),

        )

        self.linear_layers = nn.Sequential(
            nn.Linear(1792, 150),
            nn.Linear(150, 1)
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        # print(x.size())
        x = self.linear_layers(x)
        return x
    
