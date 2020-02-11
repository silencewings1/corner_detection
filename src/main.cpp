#include "alg/math_helper.h"
#include "alg/timer.h"
#include "config/config.h"
#include "detector/detector.h"
#include "detector/optimizator.h"
#include "detector/rectifier.h"
#include <future>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

void test_refy()
{
    String image_name_left = "../../imgs_1g3p_4_line_60mm/left/left_1.png";
    Mat image_left = imread(image_name_left);
    String image_name_right = "../../imgs_1g3p_4_line_60mm/right/right_1.png";
    Mat image_right = imread(image_name_right);

    Rectifier rectifier(cv::Size(1920, 1080));
    auto img_after_left = rectifier.rectify(image_left, Rectifier::LEFT);
    auto img_after_right = rectifier.rectify(image_right, Rectifier::RIGHT);

    imshow("left_ori", image_left);
    imshow("left_after", img_after_left);

    imshow("right_ori", image_right);
    imshow("right_after", img_after_right);

    imwrite("../../img_res/left_after.png", img_after_left);
    imwrite("../../img_res/right_after.png", img_after_right);

    //////////////////////////////
    String rect_image_name_left = "../../imgs_1g3p_4_line_60mm/left/left_1_rectified.png";
    Mat rect_image_left = imread(rect_image_name_left);
    String rect_image_name_right = "../../imgs_1g3p_4_line_60mm/right/right_1_rectified.png";
    Mat rect_image_right = imread(rect_image_name_right);

    cv::Mat left_sub;

    imshow("left sub", img_after_left - rect_image_left);
    imshow("right sub", img_after_right - rect_image_right);

    //waitKey(0);
}

void test_image()
{
    String image_name = "../../imgs_1g3p_4_line_60mm/left/left_1_rectified.png";
    Mat image = imread(image_name);
    if (!image.data)
    {
        printf(" No image data \n ");
        return;
    }

    Detector detector(image.size());
    auto total = tic();
    auto corners = detector.process(image);
    toc(total, "total");
    detector.showResult("corners", corners, image);

    waitKey(0);
}

void test_video_2()
{
    VideoCapture capture_left, capture_right;
    // capture_right.open("../../video20200120/WIN_20200120_11_37_50_Pro.mp4");
    // capture_left.open("../../video20200120/WIN_20200120_11_39_46_Pro.mp4");
    capture_right.open(0);
    capture_left.open(2);

    cv::Size size(1920, 1080);
    capture_left.set(CAP_PROP_FRAME_WIDTH, size.width);
    capture_left.set(CAP_PROP_FRAME_HEIGHT, size.height);
    capture_right.set(CAP_PROP_FRAME_WIDTH, size.width);
    capture_right.set(CAP_PROP_FRAME_HEIGHT, size.height);

    if (!capture_left.isOpened() || !capture_right.isOpened())
    {
        printf("can not open ...\n");
        return;
    }

    auto detector_left = std::make_unique<Detector>(size);
    auto detector_right = std::make_unique<Detector>(size);
    Optimizator optimizator;
    Rectifier rectifier(size);

    cv::namedWindow("left", WINDOW_NORMAL);
    cv::namedWindow("right", WINDOW_NORMAL);

    Mat frame_left, frame_right;

    auto rectify_avg_time = 0.0, detect_avg_time = 0.0, bad_avg_time = 0.0, good_avg_time = 0.0;
    auto count = 0, bad_count = 0, good_count = 0;
    auto total = tic();
    while (true)
    {
        if (!capture_left.read(frame_left) || !capture_right.read(frame_right))
            break;

        auto tr = tic();
        frame_left = rectifier.rectify(frame_left, Rectifier::LEFT);
        frame_right = rectifier.rectify(frame_right, Rectifier::RIGHT);
        rectify_avg_time += toc(tr, "tr");

        auto tt = tic();
#ifdef USE_MULTI_THREAD
        auto fut_left = std::async(std::launch::async, [&]() { return detector_left->process(frame_left); });
        auto fut_right = std::async(std::launch::async, [&]() { return detector_right->process(frame_right); });
        auto corners_left = fut_left.get();
        auto corners_right = fut_right.get();
#else
        auto corners_left = detector_left->process(frame_left);
        auto corners_right = detector_right->process(frame_right);
#endif // USE_MULTI_THREAD
        detect_avg_time += toc(tt, "tt");
        ++count;

        detector_left->showResult("left", corners_left, frame_left);
        detector_right->showResult("right", corners_right, frame_right);

        if (!corners_left.empty() && !corners_right.empty())
        {
            auto opti = tic();
            auto [t, cost] = optimizator.process(corners_left, corners_right);
            auto duration = toc(opti, "opti");
            if (cost >= OPTI_COST_THRESHOLD)
            {
                std::cout << "bad at: " << count << std::endl;
                bad_avg_time += duration;
                ++bad_count;
            }
            else
            {
                good_avg_time += duration;
                ++good_count;
            }
        }

        auto key = waitKey(1);
        if (key == 'q' || key == 'Q')
            break;
    }
    auto total_ = toc(total, "total");

    capture_left.release();
    capture_right.release();

    std::cout << "***************** Total Average Time: " << total_ / count << "s *****************" << std::endl;
    std::cout << "***************** Rectify Average Time: " << rectify_avg_time / count << "s *****************" << std::endl;
    std::cout << "***************** Detection Average Time: " << detect_avg_time / count << "s *****************" << std::endl;
    std::cout << "***************** Bad Average Time: " << bad_avg_time / bad_count << ", " << 100.0 * bad_count / (bad_count + good_count) << "% *****************" << std::endl;
    std::cout << "***************** Good Average Time: " << good_avg_time / good_count << ", " << 100.0 * good_count / (bad_count + good_count) << "% *****************" << std::endl;
}

void test_opti()
{
    std::vector<Corners> left, right;
    {
        // 1
        Corners left1 = {
            Corner(1316.97827148438, 525.913269042969),
            Corner(1319.77197265625, 511.513732910156),
            Corner(1321.87866210938, 497.169189453125),
            Corner(1323.06127929688, 484.185333251953),
            Corner(1313.82910156250, 539.424804687500),
            Corner(1310.02880859375, 552.004394531250),
            Corner(1306.20104980469, 563.408569335938),
            Corner(1302.06884765625, 574.021911621094)};
        Corners right1 = {
            Corner(1246.00732421875, 525.893920898438),
            Corner(1248.82177734375, 510.886199951172),
            Corner(1250.99145507813, 497.111877441406),
            Corner(1252.19958496094, 484.027923583984),
            Corner(1253.03527832031, 472.038452148438),
            Corner(1242.99475097656, 538.945800781250),
            Corner(1239.89099121094, 551.932312011719),
            Corner(1236.88757324219, 563.963012695313),
            Corner(1232.99218750000, 574.063964843750)};
        // 2
        Corners left2 = {
            Corner(1243.99108886719, 480.030090332031),
            Corner(1245.74035644531, 466.806579589844),
            Corner(1246.03649902344, 454.098754882813),
            Corner(1241.18530273438, 494.107971191406),
            Corner(1238.94995117188, 509.078186035156),
            Corner(1235.86047363281, 522.895507812500),
            Corner(1232.81188964844, 536.132019042969),
            Corner(1229.02197265625, 548.073730468750),
            Corner(1225.23352050781, 558.988098144531)};
        Corners right2 = {
            Corner(1166.07800292969, 508.978485107422),
            Corner(1168.81579589844, 493.773345947266),
            Corner(1170.94702148438, 479.841766357422),
            Corner(1172.96276855469, 466.105072021484),
            Corner(1174.07958984375, 454.113311767578),
            Corner(1162.96276855469, 522.132263183594),
            Corner(1159.97961425781, 535.956787109375),
            Corner(1156.87597656250, 547.881286621094),
            Corner(1154.18518066406, 559.001342773438)};
        // 3
        Corners left3 = {
            Corner(1154.03662109375, 506.001190185547),
            Corner(1157.22192382813, 491.881378173828),
            Corner(1159.95544433594, 476.730651855469),
            Corner(1162.73632812500, 462.041198730469),
            Corner(1164.09729003906, 448.225341796875),
            Corner(1165.91857910156, 435.906768798828),
            Corner(1151.19201660156, 519.915283203125),
            Corner(1148.09558105469, 532.101074218750),
            Corner(1145.09802246094, 543.825317382813)};
        Corners right3 = {
            Corner(1071.89794921875, 543.051452636719),
            Corner(1074.13867187500, 531.945861816406),
            Corner(1076.96899414063, 519.115600585938),
            Corner(1079.02673339844, 505.860534667969),
            Corner(1082.00305175781, 491.766845703125),
            Corner(1084.87060546875, 476.069946289063),
            Corner(1087.07287597656, 461.915466308594),
            Corner(1089.14672851563, 448.006866455078),
            Corner(1090.97558593750, 435.982635498047),
            Corner(1092.93908691406, 425.793670654297)};
        // 4
        Corners left4 = {
            Corner(1067.98742675781, 488.080810546875),
            Corner(1070.88842773438, 473.131042480469),
            Corner(1073.21948242188, 457.353881835938),
            Corner(1076.01599121094, 442.925598144531),
            Corner(1078.72448730469, 428.972717285156),
            Corner(1080.05895996094, 415.987579345703),
            Corner(1065.15429687500, 501.999847412109),
            Corner(1062.89758300781, 515.024780273438),
            Corner(1060.18627929688, 526.884460449219)};
        Corners right4 = {
            Corner(993.082153320313, 473.091094970703),
            Corner(995.889221191406, 457.710205078125),
            Corner(998.086364746094, 442.868560791016),
            Corner(1001.13891601563, 428.254791259766),
            Corner(1003.18292236328, 415.986480712891),
            Corner(990.924255371094, 487.974334716797),
            Corner(988.209228515625, 501.947692871094),
            Corner(986.220581054688, 515.012268066406),
            Corner(984.915527343750, 526.837768554688)};
        // 5
        Corners left5 = {
            Corner(979.982238769531, 454.000091552734),
            Corner(982.832885742188, 438.305572509766),
            Corner(985.123229980469, 422.918579101563),
            Corner(988.174194335938, 408.335205078125),
            Corner(990.766601562500, 395.187377929688),
            Corner(977.107116699219, 469.092193603516),
            Corner(975.060913085938, 483.891296386719),
            Corner(973.015869140625, 497.160095214844),
            Corner(971.183288574219, 509.189239501953)};
        Corners right5 = {
            Corner(900.010253906250, 453.937744140625),
            Corner(903, 439),
            Corner(905.693786621094, 422.207794189453),
            Corner(908.940734863281, 408.031036376953),
            Corner(911.104309082031, 395.129821777344),
            Corner(897.189697265625, 469.002105712891),
            Corner(896.001281738281, 483.885192871094),
            Corner(894.757934570313, 497.147766113281),
            Corner(893.921081542969, 509.078887939453)};
        // 6
        Corners left6 = {
            Corner(876.975219726563, 490.936340332031),
            Corner(877.755859375000, 478.580139160156),
            Corner(879.023071289063, 464.139312744141),
            Corner(880.890686035156, 449.052978515625),
            Corner(883.027282714844, 433.222290039063),
            Corner(885.883666992188, 416.856567382813),
            Corner(888.838134765625, 401.095275878906),
            Corner(892.060729980469, 386.227478027344),
            Corner(894.968811035156, 373.156158447266)};
        Corners right6 = {
            Corner(800.930297851563, 433.094818115234),
            Corner(804, 418),
            Corner(806.212280273438, 400.958923339844),
            Corner(810.053833007813, 386.119445800781),
            Corner(813.793884277344, 373.029815673828),
            Corner(798.877807617188, 448.960113525391),
            Corner(797.771362304688, 464.065948486328),
            Corner(796.874938964844, 478.058471679688),
            Corner(796.919616699219, 490.895446777344)};
        // 7
        Corners left7 = {
            Corner(776.985717773438, 458.121765136719),
            Corner(777.915466308594, 443.825378417969),
            Corner(778.854980468750, 427.932464599609),
            Corner(781.009094238281, 411.810699462891),
            Corner(783.789306640625, 394.715820312500),
            Corner(786.986816406250, 378.195861816406),
            Corner(790.818054199219, 363.651580810547),
            Corner(794.129821777344, 350.031555175781),
            Corner(777.120727539063, 471.132446289063)};
        Corners right7 = {
            Corner(693.998901367188, 427.902648925781),
            Corner(694, 413),
            Corner(698.139526367188, 394.124206542969),
            Corner(701.935180664063, 378.109863281250),
            Corner(706.053283691406, 363.103424072266),
            Corner(710.245605468750, 349.906219482422),
            Corner(693.778625488281, 443.235900878906),
            Corner(693.145874023438, 457.930450439453),
            Corner(694.771484375000, 471.111328125000)};
        // 8
        Corners left8 = {
            Corner(669.950073242188, 436.920562744141),
            Corner(669.964355468750, 421.681854248047),
            Corner(670.167053222656, 405.189300537109),
            Corner(672.060546875000, 388.105346679688),
            Corner(674.792907714844, 370.808715820313),
            Corner(678.665222167969, 354.103546142578),
            Corner(682.774597167969, 338.831665039063),
            Corner(687.051757812500, 325.067596435547),
            Corner(692.361755371094, 314.740417480469),
            Corner(671.054504394531, 450.101989746094)};
        Corners right8 = {
            Corner(583.951049804688, 388.050598144531),
            Corner(586.785949707031, 370.767181396484),
            Corner(590.031860351563, 353.993225097656),
            Corner(595.024719238281, 338.819671630859),
            Corner(600.228759765625, 324.921478271484),
            Corner(606.222290039063, 314.395202636719),
            Corner(582.849487304688, 405.080596923828),
            Corner(582.772827148438, 421.124114990234),
            Corner(583.071838378906, 436.233123779297),
            Corner(585.828430175781, 450.081054687500)};
        // push
        left.emplace_back(left1);
        left.emplace_back(left2);
        left.emplace_back(left3);
        left.emplace_back(left4);
        left.emplace_back(left5);
        left.emplace_back(left6);
        left.emplace_back(left7);
        left.emplace_back(left8);
        right.emplace_back(right1);
        right.emplace_back(right2);
        right.emplace_back(right3);
        right.emplace_back(right4);
        right.emplace_back(right5);
        right.emplace_back(right6);
        right.emplace_back(right7);
        right.emplace_back(right8);
    }

    Optimizator optimizator;

    std::vector<Eigen::Vector3d> centers;
    for (int i = 0; i < 8; ++i)
    {
        auto [t, cost] = optimizator.process(left.at(i), right.at(i));
        std::cout << "center(" << t(0) << ", " << t(1) << ", " << t(2) << ")" << std::endl;

        centers.push_back(t);
    }
    for (int i = 0; i < centers.size() - 1; ++i)
    {
        Eigen::Vector3d co = centers.at(i + 1) - centers.at(i);
        std::cout << "dis: " << co.norm() << std::endl;
    }
}

void test_whole()
{
    cv::Size size(1920, 1080);
    auto detector_left = std::make_unique<Detector>(size);
    auto detector_right = std::make_unique<Detector>(size);
    Rectifier rectifier(size);
    // cv::namedWindow("left", WINDOW_NORMAL);
    // cv::namedWindow("right", WINDOW_NORMAL);

    Optimizator optimizator;
    std::vector<Eigen::Vector3d> centers;

    auto normal_avg_time = 0.0;
    auto bad_avg_time = 0.0;
    auto normal_count = 0;
    auto bad_count = 0;
    for (int id = 1; id <= 16; ++id)
    {
        String name_left = "../../imgs_1g3p_4_line_60mm/left/left_" + to_string(id) + "_rectified.png";
        String name_right = "../../imgs_1g3p_4_line_60mm/right/right_" + to_string(id) + "_rectified.png";
        // String name_left = "../../imgs_1g3p_4_line_60mm/left/left_" + to_string(id) + ".png";
        // String name_right = "../../imgs_1g3p_4_line_60mm/right/right_" + to_string(id) + ".png";
        Mat image_left = imread(name_left);
        Mat image_right = imread(name_right);
        if (!image_left.data || !image_right.data)
        {
            printf(" No image data \n ");
            return;
        }

        // image_left = rectifier.rectify(image_left, Rectifier::LEFT);
        // image_right = rectifier.rectify(image_right, Rectifier::RIGHT);

        auto fut_left = std::async(std::launch::async, [&]() { return detector_left->process(image_left); });
        auto fut_right = std::async(std::launch::async, [&]() { return detector_right->process(image_right); });
        auto corners_left = fut_left.get();
        auto corners_right = fut_right.get();

        // detector_left->showResult("left", corners_left, image_left);
        // detector_right->showResult("right", corners_right, image_right);
        // waitKey(0);

        auto tt = tic();
        auto [t, cost] = optimizator.process(corners_left, corners_right);
        auto duration = toc(tt, "tt");

        centers.push_back(t);
        if (cost >= OPTI_COST_THRESHOLD)
        {
            std::cout << "id: " << id << ", bad optimization result" << std::endl;
            bad_avg_time += duration;
            ++bad_count;
        }
        else
        {
            normal_avg_time += duration;
            ++normal_count;
        }
    }

    for (int i = 0; i < centers.size() - 1; ++i)
    {
        Eigen::Vector3d co = centers.at(i + 1) - centers.at(i);
        std::cout << "dis: " << co.norm() << std::endl;
    }

    std::cout << "***************** Algorithm Normal Average Time: " << normal_avg_time / normal_count << "s *****************" << std::endl;
    std::cout << "***************** Algorithm Bad Average Time: " << bad_avg_time / bad_count << "s *****************" << std::endl;
}

int main()
{
    ////////////////
    //cv::cuda::printCudaDeviceInfo(cv::cuda::getDevice());
    ////////////////
    if (!loadConfig())
        return -1;

    // test_refy();
    test_whole();
    // test_video_2();
    waitKey(0);
    return 0;
}