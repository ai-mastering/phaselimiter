#ifndef BAKUAGE_LOF_H
#define BAKUAGE_LOF_H

#include <memory>
#include <unordered_map>
#include <vector>
#ifndef BAKUAGE_DISABLE_TBB
#include <tbb/tbb.h>
#endif
#include <boost/serialization/vector.hpp>
#include "hnswlib/hnswlib.h"
#include "bakuage/hnsw_serialization.h"

namespace boost {
    namespace serialization {
        class access;
    }
}

namespace bakuage {
    // hnswによってファイルに書き込まれるのでpackedである必要があるし、なるべく小さいと良い
    // コードを見たが、LofPointはhnswに渡したあとは破棄して良い
    // しかもPODである必要がある
    typedef int LofPoint;
    
    template <class LofSpace>
    typename LofSpace::DistType HnswDistFunc(const void *point1, const void *point2, const void *dist_func_param) {
        LofSpace *space = (LofSpace *)dist_func_param;
        return space->GetDistance((const LofPoint *)point1, (const LofPoint *)point2);
    }
    
    template <class Point, class DistT, class DistFunc>
    class LofSpace: public hnswlib::SpaceInterface<DistT> {
    public:
        typedef DistT DistType;
        
        static Point **ThreadLocalTemporaryPoint() {
            static thread_local Point *point;
            return &point;
        }
        
        LofSpace(const DistFunc &dist_func): hnswlib::SpaceInterface<DistT>(), dist_func_(dist_func), points_(nullptr) {};
        
        virtual ~LofSpace() {}
        
        DistType GetDistance(const LofPoint *point1, const LofPoint *point2) {
            // memo化をしてもメモリが増えるだけで速くならない
            return dist_func_(*point1 < 0 ? **ThreadLocalTemporaryPoint() : points_[*point1], *point2 < 0 ? **ThreadLocalTemporaryPoint() : points_[*point2]);
        }
        
        virtual hnswlib::size_t get_data_size() {
            return sizeof(LofPoint);
        }
        virtual hnswlib::DISTFUNC<DistType> get_dist_func() {
            return HnswDistFunc<LofSpace>;
        }
        virtual void *get_dist_func_param() {
            return this;
        }
        
        void set_points(const Point *value) { points_ = const_cast<Point *>(value); }
    private:
        // for boost
        friend class boost::serialization::access;
        template<class Archive>
        void serialize(Archive & ar, const unsigned int version) {
            ar & dist_func_;
        }
        
        DistFunc dist_func_;
        Point *points_;
    };

    template <class Point, class DistType, class DistFunc>
    class Lof {
    public:
        typedef hnswlib::HierarchicalNSW<DistType> Hnsw;
        typedef LofSpace<Point, DistType, DistFunc> Space;
        
        Lof(const DistFunc &dist_func): space_(dist_func), k_(0), m_(0) {}
        virtual ~Lof() {}
        
        DistType CalculateLof(const Point &point) const {
            LofPoint lof_point = -1;
            *Space::ThreadLocalTemporaryPoint() = const_cast<Point *>(&point);
            auto neighbors_queue = hnsw_->searchKnn(&lof_point, k_);
            const int neighbors_size = neighbors_queue.size();
            
            DistType dist_sum = 0;
            DistType lrd_sum = 0;
            while (neighbors_queue.size()) {
                const auto &pair = neighbors_queue.top();
                // reachable distance
                dist_sum += (std::max)(pair.first, kds_[pair.second]);
                lrd_sum += lrds_[pair.second];
                neighbors_queue.pop();
            }
            const DistType dist_mean = dist_sum / (1e-37 + neighbors_size);
            const DistType lrd = 1.0 / (1e-37 + dist_mean);
            return lrd_sum / (1e-37 + neighbors_size * lrd);
        }
        
#ifndef BAKUAGE_DISABLE_TBB
        template <class Iterator>
        void Prepare(Iterator bg, Iterator ed, int k, int m = 64) {
            // clear
            k_ = k;
            m_ = m;
            
            // prepare hnsw
            points_.clear();
            for (auto it = bg; it != ed; ++it) {
                points_.emplace_back(*it);
            }
            space_.set_points(points_.data());
            
            kds_.resize(points_.size());
            lrds_.resize(points_.size());
            
            hnsw_ = std::unique_ptr<Hnsw>(new Hnsw(&space_, points_.size(), m_, 200));
            
            for (int i = 0; i < points_.size(); i++) {
                LofPoint p = i;
                hnsw_->addPoint(&p, i); // p is not refered after this by hnsw internal
            }
            
            std::vector<std::vector<std::pair<DistType, hnswlib::labeltype>>> neighbors(points_.size()); // 大きい順
            
            // calculate neighbors
            tbb::parallel_for<int>(0, points_.size(), [this, &neighbors, k](int i) {
                LofPoint lof_point = i;
                auto neighbors_queue = hnsw_->searchKnn(&lof_point, k + 1);
                neighbors[i].resize(neighbors_queue.size() - 1);
                for (int j = 0; j < neighbors[i].size(); j++) {
                    neighbors[i][j] = neighbors_queue.top();
                    neighbors_queue.pop();
                }
                kds_[i] = neighbors[i][0].first;
            });
            
            // calculate lrd
            // https://www.slideshare.net/shogoosawa581/local-outlier-factor-75487394
            for (int i = 0; i < points_.size(); i++) {
                double sum = 0;
                for (const auto &pair: neighbors[i]) {
                    // reachable distance
                    sum += (std::max)(pair.first, kds_[pair.second]);
                }
                const double mean = sum / (1e-37 + neighbors[i].size());
                lrds_[i] = 1.0 / (1e-37 + mean);
            }
        }
#endif
    private:
        // for boost
        friend class boost::serialization::access;
        template<class Archive>
        void save(Archive & ar, const unsigned int version) const {
            ar & space_;
            ar & points_;
            ar & lrds_;
            ar & kds_;
            ar & k_;
            ar & m_;
            SaveHnswToArchive(ar, *hnsw_, version);
        }
        template<class Archive>
        void load(Archive & ar, const unsigned int version) {
            ar & space_;
            ar & points_;
            ar & lrds_;
            ar & kds_;
            ar & k_;
            ar & m_;
            space_.set_points(points_.data());
            hnsw_ = std::unique_ptr<Hnsw>(new Hnsw(&space_, points_.size(), m_, 200));
            LoadHnswFromArchive(ar, *hnsw_, version, &space_);
        }
        BOOST_SERIALIZATION_SPLIT_MEMBER()
        
        Space space_;
        std::unique_ptr<Hnsw> hnsw_;
        std::vector<Point> points_;
        bakuage::AlignedPodVector<DistType> lrds_;
        bakuage::AlignedPodVector<DistType> kds_; // k番目に近い点との距離
        int k_;
        int m_;
    };
}

#endif
