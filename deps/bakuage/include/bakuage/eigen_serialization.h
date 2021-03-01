#ifndef bakuage_eigen_serialization_h
#define bakuage_eigen_serialization_h

#include <boost/serialization/split_free.hpp>
#include <Eigen/Dense>

namespace boost {
    namespace serialization {
        
        template<class Archive, class T, int A, int B, int _Options, int _MaxRows, int _MaxCols>
        void save(Archive & ar, const Eigen::Matrix<T, A, B, _Options, _MaxRows, _MaxCols> &g, const unsigned int version) {
            int rows = g.rows();
            int cols = g.cols();
            ar << rows;
            ar << cols;
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    ar << g(i, j);
                }
            }
        }
        
        template<class Archive, class T, int A, int B, int _Options, int _MaxRows, int _MaxCols>
        void load(Archive & ar, Eigen::Matrix<T, A, B, _Options, _MaxRows, _MaxCols> &g, const unsigned int version) {
            int rows;
            int cols;
            ar >> rows;
            ar >> cols;
            g.resize(rows, cols);
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    ar >> g(i, j);
                }
            }
        }
        
        template<class Archive, class T, int A, int B, int _Options, int _MaxRows, int _MaxCols>
        inline void serialize(
                              Archive & ar,
                              Eigen::Matrix<T, A, B, _Options, _MaxRows, _MaxCols> &g,
                              const unsigned int file_version
                              ){
            split_free(ar, g, file_version);
        }
    }
}

#endif

