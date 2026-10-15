#include <opencv2/objdetect/face.hpp>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>
#include <cstdio>

static const std::string DIR = "/usr/local/share/lazy-face/";
static const std::string DB = DIR + "face.tpl";
constexpr double THRESH = 0.363;   // cosine match
constexpr int NEED = 6;            // consecutive good frames to unlock
constexpr float LIVE_MIN = 0.0f;   // anti-spoof: live score must be above this
constexpr float REAL_SIGN = 1.0f;  // set to -1.0f if real faces show NEGATIVE live values

struct Tmpl {
    float yaw = 0, pitch = 0;
    float shape[10] = {};
    cv::Mat emb;   // 1x128
};

struct Pose {
    float yaw = 0, pitch = 0;
    float shape[10] = {};
};

static void saveDB(const std::vector<Tmpl>& v) {
    std::ofstream f(DB, std::ios::binary);
    int n = (int)v.size();
    f.write((const char*)&n, sizeof(n));
    for (auto& t : v) {
        f.write((const char*)&t.yaw, sizeof(float));
        f.write((const char*)&t.pitch, sizeof(float));
        f.write((const char*)t.shape, sizeof(t.shape));
        f.write((const char*)t.emb.data, 128 * sizeof(float));
    }
}

static bool loadDB(std::vector<Tmpl>& v) {
    std::ifstream f(DB, std::ios::binary);
    int n = 0;
    if (!f.read((char*)&n, sizeof(n)) || n < 1 || n > 32) return false;
    v.clear();
    for (int i = 0; i < n; i++) {
        Tmpl t;
        t.emb.create(1, 128, CV_32F);
        f.read((char*)&t.yaw, sizeof(float));
        f.read((char*)&t.pitch, sizeof(float));
        f.read((char*)t.shape, sizeof(t.shape));
        f.read((char*)t.emb.data, 128 * sizeof(float));
        if (!f) { v.clear(); return false; }
        v.push_back(t);
    }
    return true;
}

// landmarks: 0 right eye, 1 left eye, 2 nose, 3 right mouth, 4 left mouth
static Pose poseOf(const cv::Mat& f, int i) {
    auto x = [&](int k) { return f.at<float>(i, 4 + 2 * k); };
    auto y = [&](int k) { return f.at<float>(i, 5 + 2 * k); };
    float mx = (x(0) + x(1)) * 0.5f, my = (y(0) + y(1)) * 0.5f;
    float eye = std::max(std::abs(x(1) - x(0)), 1.f);
    float mouthY = (y(3) + y(4)) * 0.5f;
    Pose p;
    p.yaw = -(x(2) - mx) / eye;   // mirrored view: + = toward screen right
    p.pitch = (y(2) - my) / std::max(mouthY - my, 1.f);
    for (int k = 0; k < 5; k++) {
        p.shape[2 * k] = (x(k) - mx) / eye;
        p.shape[2 * k + 1] = (y(k) - my) / eye;
    }
    return p;
}

// anti-spoof: crop around the face (1.5x), 128x128 RGB, returns logit difference
static float liveScore(cv::dnn::Net& net, const cv::Mat& frame, const cv::Rect& box) {
    int side = (int)(std::max(box.width, box.height) * 1.5f);
    if (side < 16) return -10.f;
    cv::Mat patch;
    cv::getRectSubPix(frame, cv::Size(side, side),
                      cv::Point2f(box.x + box.width * 0.5f, box.y + box.height * 0.5f),
                      patch);
    cv::Mat blob = cv::dnn::blobFromImage(patch, 1.0 / 255.0, cv::Size(128, 128),
                                          cv::Scalar(), true, false);
    net.setInput(blob);
    cv::Mat out = net.forward();
    return REAL_SIGN * (out.at<float>(0, 1) - out.at<float>(0, 0));
}

// headless mode for PAM: exit code 0 = recognized, 1 = anything else
static int runAuth() {
    std::vector<Tmpl> db;
    if (!loadDB(db)) { fprintf(stderr, "no face.tpl\n"); return 1;}                 // nothing enrolled = fail

    cv::VideoCapture cap(0, cv::CAP_V4L2);
    if (!cap.isOpened()) {fprintf(stderr, "camera busy/not found\n"); return 1;}
    cv::Mat frame;
    if (!cap.read(frame)) return 1;

    auto det = cv::FaceDetectorYN::create(
        DIR + "models/face_detection_yunet_2023mar.onnx", "", frame.size());
    auto rec = cv::FaceRecognizerSF::create(
        DIR + "models/face_recognition_sface_2021dec.onnx", "");
    cv::dnn::Net spoofNet = cv::dnn::readNetFromONNX(DIR + "models/best_model.onnx");

    using clk = std::chrono::steady_clock;
    auto start = clk::now();
    int ok = 0;

    while (std::chrono::duration<double>(clk::now() - start).count() < 5.0) {
        if (!cap.read(frame)) return 1;

        cv::Mat faces;
        det->detect(frame, faces);

        int best = -1;
        float bestArea = 0;
        for (int i = 0; i < faces.rows; i++) {
            float a = faces.at<float>(i, 2) * faces.at<float>(i, 3);
            if (a > bestArea) { bestArea = a; best = i; }
        }
        if (best < 0) { ok = 0; continue; }

        cv::Rect box(faces.at<float>(best, 0), faces.at<float>(best, 1),
                     faces.at<float>(best, 2), faces.at<float>(best, 3));
        cv::Mat aligned, feat;
        rec->alignCrop(frame, faces.row(best), aligned);
        rec->feature(aligned, feat);
        feat = feat.clone();
        Pose pose = poseOf(faces, best);

        const Tmpl* nearest = &db[0];
        float bd = 1e9f;
        for (auto& t : db) {
            float dy = pose.yaw - t.yaw, dp = pose.pitch - t.pitch;
            float d = dy * dy + dp * dp;
            if (d < bd) { bd = d; nearest = &t; }
        }

        double score = rec->match(feat, nearest->emb, cv::FaceRecognizerSF::FR_COSINE);
        float live = liveScore(spoofNet, frame, box);
        fprintf(stderr, "score %.2f live %+.2f\n", score, live);

        if (score >= THRESH && live >= LIVE_MIN) {
            if (++ok >= NEED) return 0;        // success
        } else {
            ok = 0;
        }
    }
    return 1;                                  // timeout = fail
}

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "auth") {
        try { return runAuth(); }
        catch(const std::exception& e) {fprintf(stderr, "error: %s\n", e.what());return 1;}
        catch (...) {fprintf(stderr, "error\n"); return 1; }              // any error = fail
    }

    cv::VideoCapture cap(0, cv::CAP_V4L2);
    if (!cap.isOpened()) return 1;
    cv::Mat frame;
    cap >> frame;

    auto det = cv::FaceDetectorYN::create(
        "models/face_detection_yunet_2023mar.onnx", "", frame.size());
    auto rec = cv::FaceRecognizerSF::create(
        "models/face_recognition_sface_2021dec.onnx", "");
    cv::dnn::Net spoofNet = cv::dnn::readNetFromONNX("models/best_model.onnx");

    std::vector<Tmpl> db;
    bool have = loadDB(db);

    const cv::Scalar NEUTRAL(255, 162, 122), OK(140, 190, 163), BAD(106, 97, 191);
    using clk = std::chrono::steady_clock;

    enum class Mode { Auth, Enroll };
    Mode mode = Mode::Auth;
    const char* steps[5] = {"look straight", "turn left", "turn right",
                            "look up", "look down"};
    int step = 0, hold = 0, ok = 0;
    float centerPitch = 0;
    cv::Mat acc;
    std::vector<Tmpl> fresh;
    bool unlocked = false;
    auto tUnlock = clk::now();

    while (cap.read(frame)) {
        cv::Mat faces;
        det->detect(frame, faces);

        int best = -1;
        float bestArea = 0;
        for (int i = 0; i < faces.rows; i++) {
            float a = faces.at<float>(i, 2) * faces.at<float>(i, 3);
            if (a > bestArea) { bestArea = a; best = i; }
        }

        bool face = best >= 0;
        cv::Mat feat;
        cv::Rect box;
        Pose pose{};
        if (face) {
            box = cv::Rect(faces.at<float>(best, 0), faces.at<float>(best, 1),
                           faces.at<float>(best, 2), faces.at<float>(best, 3));
            cv::Mat aligned;
            rec->alignCrop(frame, faces.row(best), aligned);
            rec->feature(aligned, feat);
            feat = feat.clone();
            pose = poseOf(faces, best);
        }

        std::string msg;
        cv::Scalar color = NEUTRAL;
        double score = -1;
        float live = 0;

        if (mode == Mode::Enroll) {
            if (!face) {
                hold = 0;
                acc.release();
                msg = cv::format("%s  (%d/5)", steps[step], step + 1);
            } else {
                bool inPose = false;
                switch (step) {
                    case 0: inPose = std::abs(pose.yaw) < 0.08f; break;
                    case 1: inPose = pose.yaw < -0.18f; break;
                    case 2: inPose = pose.yaw > 0.18f; break;
                    case 3: inPose = pose.pitch - centerPitch < -0.06f && std::abs(pose.yaw) < 0.12f; break;
                    case 4: inPose = pose.pitch - centerPitch > 0.06f && std::abs(pose.yaw) < 0.12f; break;
                }
                msg = cv::format("%s  (%d/5)", steps[step], step + 1);
                if (inPose) {
                    color = OK;
                    if (acc.empty()) acc = cv::Mat::zeros(1, 128, CV_32F);
                    acc += feat;
                    if (++hold >= 6) {
                        Tmpl t;
                        t.yaw = pose.yaw;
                        t.pitch = pose.pitch;
                        std::copy(pose.shape, pose.shape + 10, t.shape);
                        t.emb = acc / (float)hold;
                        fresh.push_back(t);
                        if (step == 0) centerPitch = pose.pitch;
                        acc.release();
                        hold = 0;
                        if (++step >= 5) {
                            db = fresh;
                            saveDB(db);
                            have = true;
                            mode = Mode::Auth;
                            ok = 0;
                        }
                    }
                } else {
                    hold = 0;
                    acc.release();
                }
            }
        } else {
            if (unlocked) {
                msg = "UNLOCKED";
                color = OK;
                if (std::chrono::duration<double>(clk::now() - tUnlock).count() > 1.5) {
                    unlocked = false;
                    ok = 0;
                }
            } else if (!have) {
                msg = "press E to enroll";
            } else if (!face) {
                ok = 0;
                msg = "look at the camera";
            } else {
                // pick the enrolled template closest to the current angle
                const Tmpl* nearest = &db[0];
                float bd = 1e9f;
                for (auto& t : db) {
                    float dy = pose.yaw - t.yaw, dp = pose.pitch - t.pitch;
                    float d = dy * dy + dp * dp;
                    if (d < bd) { bd = d; nearest = &t; }
                }
                score = rec->match(feat, nearest->emb, cv::FaceRecognizerSF::FR_COSINE);
                live = liveScore(spoofNet, frame, box);

                if (score < THRESH) {
                    ok = 0;
                    color = BAD;
                    msg = "unknown";
                } else if (live < LIVE_MIN) {
                    ok = 0;
                    color = BAD;
                    msg = "spoof detected";
                } else {
                    color = OK;
                    msg = "scanning...";
                    if (++ok >= NEED) { unlocked = true; tUnlock = clk::now(); }
                }
            }
        }

        // display a mirrored copy; detection and matching use the raw frame
        cv::Mat view;
        cv::flip(frame, view, 1);
        if (face) {
            cv::Rect mb(frame.cols - box.x - box.width, box.y, box.width, box.height);
            cv::rectangle(view, mb, color, 2);
        }
        cv::putText(view, msg, {20, 40}, cv::FONT_HERSHEY_SIMPLEX, 0.9, color, 2);
        cv::putText(view,
                    cv::format("score %.2f  live %+.2f  yaw %+.2f  pitch %.2f",
                               score, live, pose.yaw, pose.pitch),
                    {20, view.rows - 15}, cv::FONT_HERSHEY_SIMPLEX, 0.45, NEUTRAL, 1);
        cv::imshow("lazy-face", view);

        int k = cv::waitKey(1);
        if (k == 27) break;
        if ((k == 'e' || k == 'E') && mode == Mode::Auth && face) {
            mode = Mode::Enroll;
            step = 0;
            hold = 0;
            fresh.clear();
            acc.release();
            unlocked = false;
        }
    }
}
