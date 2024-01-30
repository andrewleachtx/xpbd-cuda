#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <stdexcept>

namespace apbd {

template <size_t N> class NVec {
public:
  float data[N];

  NVec() : data() {}
  NVec(std::initializer_list<float> l) : data() {
    if (l.size() != N) {
      throw std::invalid_argument("Initializer has incorrect size!");
    }
    size_t i = 0;
    for (float x : l) {
      (*this)[i] = x;
      i++;
    }
  }
  float &operator[](size_t i) { return this->data[i]; }

  NVec<N> &operator+=(NVec<N> &b) {
    for (size_t i = 0; i < N; i++) {
      (*this)[i] += b[i];
    }
    return *this;
  }

  NVec<N> &operator/=(NVec<N> &b) {
    for (size_t i = 0; i < N; i++) {
      (*this)[i] /= b[i];
    }
    return *this;
  }

  NVec<N> &operator/=(float b) {
    for (size_t i = 0; i < N; i++) {
      (*this)[i] += b;
    }
    return *this;
  }

  void zero() {
    for (size_t i = 0; i < N; i++) {
      (*this)[i] = 0;
    }
  }

  template <size_t M> NVec<M> get(size_t start, size_t end) {
    NVec<M> out;
    for (size_t i = start; i < end; i++) {
      out[i] = (*this)[i];
    }
    return out;
  }

  template <size_t M> void set(size_t start, size_t end, NVec<M> &other) {
    for (size_t i = 0; i < M && i + start < end; i++) {
      (*this)[i + start] = other[i];
    }
  }

  float norm() {
    float total = 0.0;
    for (size_t i = 0; i < N; i++) {
      total += (*this)[i] * (*this)[i];
    }
    return std::sqrt(total);
  }
};

} // namespace apbd
