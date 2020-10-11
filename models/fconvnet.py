import torch.nn as nn


class fconvnet1(nn.Module):                                             # As decribed by the Fconv paper
    def __init__(self):
        super(fconvnet1, self).__init__()

        ##### STEP 1.   three parallel cnns with kernels of 1, 3, 5 resulting in 16 channels each
        self.cnn_layer_one = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.cnn_layer_two = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.cnn_layer_three = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True),
        )

        ##### STEP 2.  takes a 126 x 32 image with one channel applies convolution with custom intialized filers with 1 x 32 and extends to 150 channels
        self.real_fconv = nn.Conv2d(1, 150, kernel_size = (1, 32), padding = 0)
        self.imag_fconv = nn.Conv2d(1, 150, kernel_size = (1, 32), padding = 0)

        real_W = initialize_real(150, 32)
        imag_W = initialize_imag(150, 32)

        self.real_fconv.weight = torch.nn.Parameter(real_W)
        self.imag_fconv.weight = torch.nn.Parameter(imag_W)

        self.batch_norm = nn.BatchNorm2d(150)
        
        
        ###### STEP 3.  takes in a 1 x 126 x 32 image and performs a dilated colvolution with kernel size of 3 and dilation of 2 and extends to 16 channels  
        self.dilated_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding = 2, stride = 1, dilation = 2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True),
        )


        ##### STEP 4

        self.final_convs = nn.Sequential(
            nn.Conv2d(16, 128, kernel_size=(3, 3), padding = 1, stride = 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(128, 128, kernel_size=(3, 3), padding = 1, stride = 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(128, 128, kernel_size=(3, 3), padding = 1, stride = 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2),

        )

        self.global_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size = (15, 18))
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(128, 256),
            nn.Linear(256, 1)
        )
        
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    # Defining the forward pass    
    def forward(self, x):
        x_1_real = self.cnn_layer_one(x)
        x_2_real = self.cnn_layer_two(x)
        x_3_real = self.cnn_layer_three(x)
        x = torch.cat((x_1_real, x_2_real, x_3_real), 1)
        x = torch.sum(x, dim=1)
        x = x.unsqueeze(1)

        real = self.relu(self.batch_norm(self.real_fconv(x)))

        imag = self.relu(self.batch_norm(self.imag_fconv(x)))

        x = real + imag

        x = x.transpose_(1, 3)

        x = self.dilated_conv(x)

        x = self.final_convs(x)

        x = self.global_pool(x)

        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)

        return x.reshape(-1, 1)



class fconvnet2(nn.Module):                                       # working with two initialized kernels that function as FTFS and then using preprocessing step to return into real
    def __init__(self):
        super(fconvnet2, self).__init__()
      
        ##### STEP 1.   three parallel cnns with kernels of 1, 3, 5 resulting in 16 channels each
        self.cnn_layer_one = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.cnn_layer_two = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.cnn_layer_three = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True),
        )

        ##### STEP 2.  takes a 126 x 32 image with one channel applies convolution with custom intialized filers with 1 x 32 and extends to 150 channels
        self.real_fconv = nn.Conv2d(1, 126, kernel_size = (1, 32), padding = 0)
        self.imag_fconv = nn.Conv2d(1, 126, kernel_size = (1, 32), padding = 0)

        real_W = initialize_real(126, 32)
        imag_W = initialize_imag(126, 32)

        self.real_fconv.weight = torch.nn.Parameter(real_W)
        self.imag_fconv.weight = torch.nn.Parameter(imag_W)

        self.batch_norm = nn.BatchNorm2d(126)
        
        
        ###### STEP 3.  takes in a 1 x 126 x 32 image and performs a dilated colvolution with kernel size of 3 and dilation of 2 and extends to 16 channels  
        self.dilated_conv = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding = 2, stride = 1, dilation = 2),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.1, inplace=True),
        )


        ##### STEP 4

        self.final_convs = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=(3, 3), padding = 1, stride = 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(16, 32, kernel_size=(3, 3), padding = 1, stride = 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2),
                        
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding = 1, stride = 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 64, kernel_size=(3, 3), padding = 1, stride = 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 64, kernel_size=(3, 3), padding = 1, stride = 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2),

        )

        self.global_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size = (15, 18))
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(576, 256),
            nn.Linear(256, 1)
        )
        
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    # Defining the forward pass    
    def forward(self, x):
        x_1_real = self.cnn_layer_one(x)
        x_2_real = self.cnn_layer_two(x)
        x_3_real = self.cnn_layer_three(x)
        x = torch.cat((x_1_real, x_2_real, x_3_real), 1)
        x = torch.sum(x, dim=1)
        x = x.unsqueeze(1)
        real = self.relu(self.batch_norm(self.real_fconv(x)))
        imag = self.relu(self.batch_norm(self.imag_fconv(x)))
        x = real + imag
        x = x.transpose_(1, 3)
        x = self.dilated_conv(x)
        x = self.final_convs(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)

        return x.reshape(-1, 1)



class fconvnet3(nn.Module):            # breaking into imaginary and real with FTFS and working with coefficients of complex values
    def __init__(self):
        super(fconvnet3, self).__init__()
        

        self.stft = STFT(
            filter_length=100, 
            hop_length=50, 
            win_length=32,
            window=32).to(device)


        self.cnn_layer_one = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.cnn_layer_two = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.cnn_layer_three = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True),
        )


        self.real_fconv = nn.Conv2d(1, 1, kernel_size = 3, padding = 1)
        self.imag_fconv = nn.Conv2d(1, 1, kernel_size = 3, padding = 1)

        real_W = initialize_real_two(3, 3)
        imag_W = initialize_imag_two(3, 3)

        self.real_fconv.weight = torch.nn.Parameter(real_W)
        self.imag_fconv.weight = torch.nn.Parameter(imag_W)


        self.dilated_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding = 2, stride = 1, dilation = 2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.final_convs = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=(3, 3), padding = 1, stride = 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(16, 32, kernel_size=(3, 3), padding = 1, stride = 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=(8, 3), padding = 1, stride = 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2),

        )

        self.global_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size = (13, 4))
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(3136, 256),
            nn.Linear(256, 1)
        )


        

    # Defining the forward pass    
    def forward(self, x_real):
        x_1_real = self.cnn_layer_one(x_real)
        x_2_real = self.cnn_layer_two(x_real)
        x_3_real = self.cnn_layer_three(x_real)
        x_real = torch.cat((x_1_real, x_2_real, x_3_real), 1)
        x_real = x_real.transpose_(1, 2).transpose_(2, 3)
        x_real = x_real.reshape(list(x_real.size())[0], list(x_real.size())[1], 1, -1)
        x_real = x_real.squeeze(2)
        magnitude = []
        phase = []
        for i in range(list(x_real.size())[0]):
          temp_x = x_real[i]
          temp_magnitude, temp_phase = self.stft.transform(x_real[i])
          magnitude.append(temp_magnitude)
          phase.append(temp_phase)
        magnitude = torch.stack(magnitude, 0)
        phase = torch.stack(phase, 0)
        real = magnitude.reshape(list(magnitude.size())[0], list(magnitude.size())[1], -1, 1)
        imag = phase.reshape(list(phase.size())[0], list(phase.size())[1], -1, 1)
        real = real.transpose_(1, 3).transpose_(2, 3)
        imag = imag.transpose_(1, 3).transpose_(2, 3)

        real = self.real_fconv(real)
        imag = self.imag_fconv(imag)
        x = real + imag
        x = self.dilated_conv(x)
        x = self.final_convs(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)

        return x.reshape(-1, 1)


        
class fconvnet4(nn.Module):                                       # working with two initialized kernels that function as FTFS and then using preprocessing step to return into real
    def __init__(self):
        super(fconvnet4, self).__init__()
      
        self.real_fconv = nn.Conv2d(1, 126, kernel_size = (1, 32), padding = 0)
        self.imag_fconv = nn.Conv2d(1, 126, kernel_size = (1, 32), padding = 0)

        real_W = initialize_real(126, 32)
        imag_W = initialize_imag(126, 32)

        self.real_fconv.weight = torch.nn.Parameter(real_W)
        self.imag_fconv.weight = torch.nn.Parameter(imag_W)

        self.batch_norm = nn.BatchNorm2d(126)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
 
        self.dilated_conv = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding = 2, stride = 1, dilation = 2),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size = 2)
        )


        ##### STEP 4

        self.final_convs = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=(3, 3), padding = 1, stride = 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=3),

            nn.Conv2d(16, 32, kernel_size=(3, 3), padding = 1, stride = 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=3),
                        
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding = 1, stride = 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=3, padding = 1),

        )


        self.linear_layers = nn.Sequential(
            nn.Linear(576, 100),
            nn.Linear(100, 1)
        )
        

    # Defining the forward pass    
    def forward(self, x):
        real = self.relu(self.batch_norm(self.real_fconv(x)))
        imag = self.relu(self.batch_norm(self.imag_fconv(x)))
        x = real + imag
        x = x.transpose_(1, 3)
        x = self.dilated_conv(x)
        x = self.final_convs(x)
        x = x.view(x.size(0), -1)
        # print(x.size())
        x = self.linear_layers(x)
        return x.reshape(-1, 1)




class fconvnet5(nn.Module):                                       # working with two initialized kernels that function as FTFS and then using preprocessing step to return into real
    def __init__(self):
        super(fconvnet5, self).__init__()
      
        self.real_fconv = nn.Conv2d(1, 126, kernel_size = (1, 32), padding = 0)
        self.imag_fconv = nn.Conv2d(1, 126, kernel_size = (1, 32), padding = 0)

        real_W = initialize_real(126, 32)
        imag_W = initialize_imag(126, 32)

        self.real_fconv.weight = torch.nn.Parameter(real_W)
        self.imag_fconv.weight = torch.nn.Parameter(imag_W)

        self.batch_norm = nn.BatchNorm2d(126)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
 

        ##### STEP 4

        self.final_convs = nn.Sequential(

            # Layer 1
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2),

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
            nn.Linear(128, 64),
            nn.Linear(64, 1)
        )
        

    # Defining the forward pass    
    def forward(self, x):
        real = self.relu(self.batch_norm(self.real_fconv(x)))
        imag = self.relu(self.batch_norm(self.imag_fconv(x)))
        x = real + imag
        x = x.transpose_(1, 3)
        x = self.final_convs(x)
        x = x.view(x.size(0), -1)
        # print(x.size())
        x = self.linear_layers(x)
        return x.reshape(-1, 1)



