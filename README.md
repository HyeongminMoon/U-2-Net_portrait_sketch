U2Net_portrait_sketchy
----------------------

If you want to get information about U2Net, please refer [this original repository](https://github.com/xuebinqin/U-2-Net)

Based on segmetation model U-2-Net, we discussed the possibility of using U-2-Net to draw portrait.
It made quite good results, but we need to enhance it so we trained model by custom dataset & image pre-processing, etc.

**I hope that this described methods would be helpful to you😀**

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
[original repository](https://github.com/xuebinqin/U-2-Net) proposed using of portrait sketchy with [APDrawingGAN dataset](https://github.com/yiranran/APDrawingGAN), it is quite good for drawing **portrait only** but still it has bad performance, especially when it draws landscape.

![748066F02_1-17-TH-05-975_00310](https://user-images.githubusercontent.com/32811724/143389405-242a56ce-f099-4c4a-b1c0-473db3b138f7.png)

So we enhanced it by using custom dataset and dataset transforms, such as random bluring, random affine, jittering head, jittering nearby human's outline.

![750365F02_1-17-TH-09-374_00106](https://user-images.githubusercontent.com/32811724/143390222-ddcb5f1d-dea0-45ff-9fba-5a614bcf9f46.png)

But It still has many dirty messy pixels on the whole picture, we want to make it clear so applied Opencv based morphology, filling contour, blurring etc.

![744359F02_1-17-TH-04-780_00594](https://user-images.githubusercontent.com/32811724/143390737-fadb6c55-101c-49da-8d68-6c62c501cc00.png)

It becomes quite good, but we continued because we want to make it lose more information such as face&cloth detail, but keep the important line information.

![758050F02_1-18-TH-03-295_00425](https://user-images.githubusercontent.com/32811724/143391038-b7fc3ddd-53c4-4071-bd5a-ae3077e69485.png)

