#include <gtest/gtest.h>
#include <set>
#include "reconstruction/psd_estimator.h"

template<typename T>
class PSD_Estimator_Test : public ::testing::Test {
public:
    void testHalf2Whole(size_t x, size_t y) {
        // allocate data
        auto settings = FFTSettings<T>(x, y);
        auto in = new T[settings.fElemsBatch()];
        auto out = new T[settings.sElemsBatch()];

        // fill it
        for (size_t n = 0; n < settings.fElemsBatch(); ++n) {
            in[n] = n;
        }

//        printf("in:\n");
//        for(size_t y = 0; y < settings.fDim().y(); ++y) {
//            for(size_t x = 0; x < settings.fDim().x(); ++x) {
//                size_t index = y * settings.fDim().x() + x;
//                printf("%5.1f ", in[index]);
//            }
//            printf("\n");
//        }
//        printf("\n");

        // call
        PSDEstimator<T>::half2whole(in, out, settings, [&](bool mirror, T val){return val;});

//        printf("out:\n");
//        for(size_t y = 0; y < settings.sDim().y(); ++y) {
//            for(size_t x = 0; x < settings.sDim().x(); ++x) {
//                size_t index = y * settings.sDim().x() + x;
//                printf("%5.1f ", out[index]);
//            }
//            printf("\n");
//        }
//        printf("\n");

        // test 'original' half
        for (size_t y = 0; y < settings.sDim().y(); ++y) {
            for (size_t x = 0; x < settings.fDim().x(); ++x) {
                auto indexIn = y * settings.fDim().x() + x;
                auto indexOut = y * settings.sDim().x() + x;
                ASSERT_EQ(in[indexIn], out[indexOut]);
            }
        }

        // test 'new' half
        for (size_t y = 0; y < settings.sDim().y(); ++y) {
            for (size_t x = 0; x < (settings.sDim().x() - settings.fDim().x()); ++x) {
                size_t xIn = x + 1;
                size_t yIn = (settings.sDim().y() - y) % settings.sDim().y();
                size_t xOut = settings.sDim().x() - x - 1;
                auto indexIn = yIn * settings.fDim().x() + xIn;
                auto indexOut = y * settings.sDim().x() + xOut;
                ASSERT_EQ(in[indexIn], out[indexOut]);
            }
        }

        delete[] in;
        delete[] out;
    }


    void test_window(const std::pair<size_t, size_t> &borders,
            const Dimensions &micrograph,
            const Dimensions &patch,
            float overlap) {

//        printf("testing %lu %lu -> %lu %lu, borders %lu %lu, overlap %f\n",
//                micrograph.x(), micrograph.y(),
//                patch.x(), patch.y(),
//                borders.first, borders.second,
//                overlap);

        // run it first to catch possible problems
        const auto result = PSDEstimator<T>::getPatchesLocation(borders, micrograph, patch, overlap);

        auto sizesX = std::set<size_t>();
        auto sizesY = std::set<size_t>();

        size_t stepX = std::max((1.f - overlap) * patch.x(), 1.f);
        size_t stepY = std::max((1.f - overlap) * patch.y(), 1.f);

        size_t div_NumberX = ceil(micrograph.x() / (float)stepX);
        size_t div_NumberY = ceil(micrograph.y() / (float)stepY);
        size_t div_Number = div_NumberX * div_NumberY;
        for (size_t n = 0; n < div_Number; ++n) {
            bool skip = false;
            size_t blocki = n / div_NumberX;
            size_t blockj = n % div_NumberX;

            size_t y = borders.second + blocki * stepY;
            size_t x = borders.first + blockj * stepX;

            // test if the full piece is inside the micrograph
            if (y + patch.y() > (micrograph.y() - borders.second))
                y = micrograph.y() - patch.y() - borders.second;
            if (x + patch.x() > (micrograph.x() - borders.first))
                x = micrograph.x() - patch.x() - borders.first;
            // store sizes
            sizesX.emplace(x);
            sizesY.emplace(y);
        }

//        printf("candidX: ");
//        for (auto x : sizesX) printf("%lu ", x);
//        printf("\ncandidY: ");
//        for (auto y : sizesY) printf("%lu ", y);
//        printf("\n");

        // test that we have right number of results
        EXPECT_EQ(sizesX.size() * sizesY.size(), result.size());

        // test that we have proper sizes
        for (auto &r : result) {
            bool isXIn = std::find(sizesX.begin(), sizesX.end(), r.tl.x) != sizesX.end();
            bool isYIn = std::find(sizesY.begin(), sizesY.end(), r.tl.y) != sizesY.end();
            size_t sizeX = r.br.x - r.tl.x + 1;
            size_t sizeY = r.br.y - r.tl.y + 1;

            ASSERT_TRUE(isXIn) << r.tl.x;
            ASSERT_TRUE(isYIn) << r.tl.y;
            ASSERT_EQ(patch.x(), sizeX);
            ASSERT_EQ(patch.y(), sizeY);
            ASSERT_LT(r.br.x, micrograph.x());
            ASSERT_LT(r.br.y, micrograph.y());
        }
    }
};
TYPED_TEST_SUITE_P(PSD_Estimator_Test);

TYPED_TEST_P( PSD_Estimator_Test, windowCoords)
{
    auto inXCandid = {32, 256, 512};
    auto inYCandid = {32, 256, 512, 513};
    auto patchXCandid = {5, 64, 367, 512};
    auto patchYCandid = {5, 64, 367, 512};
    auto borderXCandid = {0, 5};
    auto borderYCandid = {0, 5};
    auto overlapCandid = {0.f, 0.2f, 0.9f};

    int counter = 0;
    for (auto inX : inXCandid) {
        for (auto inY : inYCandid) {
            for (auto patchX : patchXCandid) {
                if (patchX > inX) continue;
                for (auto patchY : patchYCandid) {
                    if (patchY > inY) continue;
                    for (auto borderX : borderXCandid) {
                        if (patchX + 2 * borderX > inX) continue;
                        for (auto borderY : borderYCandid) {
                            if (patchY + 2 * borderY > inY) continue;
                            counter = (counter + 1) % overlapCandid.size();
                            PSD_Estimator_Test<TypeParam>::test_window(
                                {borderX, borderY},
                                Dimensions(inX, inY),
                                Dimensions(patchX, patchY),
                                *(overlapCandid.begin() + counter));
                        }
                    }
                }
            }
        }
    }
}

TYPED_TEST_P( PSD_Estimator_Test, half2whole)
{
    // even even (small)
    PSD_Estimator_Test<TypeParam>::testHalf2Whole(10, 10);
    PSD_Estimator_Test<TypeParam>::testHalf2Whole(4, 10);
    // odd even (small)
    PSD_Estimator_Test<TypeParam>::testHalf2Whole(11, 10);
    // even odd (small)
    PSD_Estimator_Test<TypeParam>::testHalf2Whole(10, 11);
    // odd odd (small)
    PSD_Estimator_Test<TypeParam>::testHalf2Whole(11, 11);
    PSD_Estimator_Test<TypeParam>::testHalf2Whole(5, 11);

    // even even (big)
    PSD_Estimator_Test<TypeParam>::testHalf2Whole(4096, 4096);
    PSD_Estimator_Test<TypeParam>::testHalf2Whole(128, 4096);
    // odd even (big)
    PSD_Estimator_Test<TypeParam>::testHalf2Whole(4097, 4096);
    // even odd (big)
    PSD_Estimator_Test<TypeParam>::testHalf2Whole(4096, 4097);
    // odd odd (big)
    PSD_Estimator_Test<TypeParam>::testHalf2Whole(4095, 4097);
    PSD_Estimator_Test<TypeParam>::testHalf2Whole(127, 4097);
}


REGISTER_TYPED_TEST_SUITE_P(PSD_Estimator_Test,
    windowCoords,
    half2whole
);

typedef ::testing::Types<double, float> TestTypes;
INSTANTIATE_TYPED_TEST_SUITE_P(psd_estimator, PSD_Estimator_Test, TestTypes);
