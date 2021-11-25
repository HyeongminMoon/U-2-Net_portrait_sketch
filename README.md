U2Net_portrait_sketchy
----------------------

If you want to get information about U2Net, please refer [this original repository](https://github.com/xuebinqin/U-2-Net)

Based on segmetation model U-2-Net, we discussed the possibility of using U-2-Net to draw portrait.
It made quite good results, but we need to enhance it so we trained model by custom dataset & image pre-processing, etc.

**Sorry for we cant upload trained model and custom dataset, I hope that this described methods would be helpful to you**

## Usage

1. Various styles of portrait sketchy

<br><img src="https://user-images.githubusercontent.com/32811724/143386093-f9f3b1e0-4e8b-4fcd-9303-56a872888f5d.png" width="320px" height="180px"></img>
<img src="https://user-images.githubusercontent.com/32811724/143386103-8eb3fb3b-8bed-4f37-8a49-3b4ef1718fe4.png" width="320px" height="180px"></img>
</br>
<br><img src="https://user-images.githubusercontent.com/32811724/143386178-d2eeb72e-2a33-43c8-b0f8-713efbf30709.png" width="320px" height="180px"></img>
<img src="https://user-images.githubusercontent.com/32811724/143386194-614bfe30-e025-47b7-a899-43839344c172.png" width="320px" height="180px"></img>
</br>
<br><img src="https://user-images.githubusercontent.com/32811724/143386187-71006f1b-9e29-4158-b01b-f4541ad057f4.png" width="320px" height="180px"></img>
<img src="https://user-images.githubusercontent.com/32811724/143386196-e7215ff1-7b2c-4e9c-8554-4bd8c94fc3ff.png" width="320px" height="180px"></img>
</br>

2. Video Application

![IU_Coin_train_custom_erode2 (1)](https://user-images.githubusercontent.com/32811724/143388489-1d9e0756-58e9-4ab6-98f9-ca0f0c044869.gif)
![Traffic_light_custom_b2](https://user-images.githubusercontent.com/32811724/143388497-3237d2db-3b80-4309-97f3-a4d7aae28321.gif)

## Enhancement of portrait sketchy
[original repository](https://github.com/xuebinqin/U-2-Net) proposed using of portrait sketchy with [APDrawingGAN dataset](https://github.com/yiranran/APDrawingGAN), it is quite good for drawing **portrait only** but still it has bad performance, especially when it draw landscape.
![748066F02_1-17-TH-05-975_00310](https://user-images.githubusercontent.com/32811724/143389405-242a56ce-f099-4c4a-b1c0-473db3b138f7.png)
