## 图像校正

把`rectifier.h`放到`detector`目录下

以下为测试代码

```c++
#include "detector/rectifier.h"

void test_recrify()
{
    cv::String image_name = "../../imgs_1g3p_4_line_60mm/left/left_1.png";
	cv::Mat image = imread(image_name);
	if (!image.data)
	{
		printf(" No image data \n ");
		return;
	}

	Rectifier rectifier(cv::Size(1920, 1080));
	auto img_rectified = rectifier.rectify(image, Rectifier::LEFT);
    
    cv::imshow("img_rectified", img_rectified);
    
    // Detector detector(img_rectified);
}
```

