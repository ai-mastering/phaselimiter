#include "gtest/gtest.h"
#include <limits>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include "bakuage/lof.h"
#include "bakuage/memory.h"
#include "bakuage/vector_math.h"

namespace {
    typedef bakuage::AlignedPodVector<double> Point;
    
    struct DistFunc {
        double operator () (const Point &a, const Point &b) {
            return bakuage::VectorNormDiffL2(a.data(), b.data(), a.size());
        }
        
    private:
        friend class boost::serialization::access;
        template<class Archive>
        void serialize(Archive & ar, const unsigned int version) {}
    };
}

TEST(Lof, SameAsScikit) {
    const int dim = 2;
    const std::vector<double> points_data({
        0, 1,
        2, 3,
        4, 5,
        16, 17,
    });
    const std::vector<double> query_data({
        -2, -1,
        1, 0,
        3, 2,
        20, 10,
    });
    const std::vector<double> expected({
        1.16666667,
        0.90942658,
        0.875,
        2.65161739,
    });
    
    DistFunc func;
    bakuage::Lof<Point, double, DistFunc> lof(func);
    std::vector<Point> points;
    std::vector<Point> querys;
    for (int i = 0; i < points_data.size() / dim; i++) {
        Point v(dim);
        for (int j = 0; j < dim; j++) {
            v[j] = points_data[dim * i + j];
        }
        points.emplace_back(v);
    }
    for (int i = 0; i < query_data.size() / dim; i++) {
        Point v(dim);
        for (int j = 0; j < dim; j++) {
            v[j] = query_data[dim * i + j];
        }
        querys.emplace_back(v);
    }
    lof.Prepare(points.begin(), points.end(), 2);
    
    for (int i = 0; i < expected.size(); i++) {
        const auto x = lof.CalculateLof(querys[i]);
        EXPECT_NEAR(expected[i], x, 1e-7);
    }
}

TEST(Lof, Serialization) {
    const int dim = 2;
    const std::vector<double> points_data({
        0, 1,
        2, 3,
        4, 5,
        16, 17,
    });
    const std::vector<double> query_data({
        -2, -1,
        1, 0,
        3, 2,
        20, 10,
    });
    const std::vector<double> expected({
        1.16666667,
        0.90942658,
        0.875,
        2.65161739,
    });
    
    std::stringstream ss;
    {
        std::vector<Point> points;
        for (int i = 0; i < points_data.size() / dim; i++) {
            Point v(dim);
            for (int j = 0; j < dim; j++) {
                v[j] = points_data[dim * i + j];
            }
            points.emplace_back(v);
        }
        DistFunc func;
        bakuage::Lof<Point, double, DistFunc> lof(func);
        lof.Prepare(points.begin(), points.end(), 2);
        boost::archive::binary_oarchive oa(ss);
        oa << lof;
    }
    
    DistFunc func;
    bakuage::Lof<Point, double, DistFunc> lof2(func);
    {
        boost::archive::binary_iarchive ia(ss);
        ia >> lof2;
    }
    
    std::vector<Point> querys;
    for (int i = 0; i < query_data.size() / dim; i++) {
        Point v(dim);
        for (int j = 0; j < dim; j++) {
            v[j] = query_data[dim * i + j];
        }
        querys.emplace_back(v);
    }
    
    for (int i = 0; i < expected.size(); i++) {
        const auto x = lof2.CalculateLof(querys[i]);
        EXPECT_NEAR(expected[i], x, 1e-7);
    }
}
