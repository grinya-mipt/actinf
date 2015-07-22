#ifndef __VECTOR_H
#define __VECTOR_H
/**
 * \file vector.h
 *
 * Defines the util::Vector class template
 */

#include <iostream>
#include <cmath>
#include <cassert>
#include <cstdarg>
#include <util/static_assert.h>
#include <storage/fwd.h>

/**
 * \brief Namespace containing all the utility classes.
 */
namespace util
{
  /**
   * \class Vector vector.h <util/vector.h>
   *
   * Vector class supporting all classic classic vector operations.
   *
   * \note The addition and subtraction with a scalar is possible and is 
   * equivalent to adding/subtracting a vector filled with the scalar.
   */
  template <size_t dim,class T = double>
    class Vector
  {
  protected:
    T elems[dim];

  public:
    typedef T value_type;
    typedef T& reference_type;
    typedef const T& const_reference_type;
    typedef T* pointer_type;
    typedef const T* const_pointer_type;
    typedef T* iterator;
    typedef const T* const_iterator;

    /**
     * Default constructor
     */
    __device__ __host__
    Vector() {}

    /**
     * Copy another vector
     */
    __device__ __host__
    Vector(const Vector& vec)
    {
      for(size_t i = 0 ; i < dim ; i++)
        elems[i] = vec[i];
    }

    /**
     * Copy another vector with different number of elements
     */
    template <size_t d1, class T1>
    __device__ __host__
    Vector(const Vector<d1,T1>& vec)
    {
      if(d1 < dim)
      {
        for(size_t i = 0 ; i < d1 ; ++i)
          elems[i] = vec[i];
        for(size_t i = d1 ; i < dim ; ++i)
          elems[i] = 0;
      }
      else
      {
        for(size_t i = 0 ; i < dim ; ++i)
          elems[i] = vec[i];
      }
    }

    /**
     * Initialize a vector from any object behaving like an array.
     *
     * The only constraints are:
     *  - the type has to be convertible to \c T
     *  - the vector needs a [] operator
     *  - the size of the vector has to be at least \c dim
     */
    template <class Vec>
    __device__ __host__
    Vector(const Vec& el)
    {
      for(size_t i = 0 ; i < dim ; i++)
        elems[i] = el[i];
    }

    /**
     * Initialize a vector with all values to \c x.
     */
    __device__ __host__
    Vector( const T& x )
    {
      for( size_t i = 0 ; i < dim ; ++i )
        elems[ i ] = x;
    }

    /**
     * Initialize a 2D vector
     */
    __device__ __host__
    Vector( const T& x, const T& y )
    {
      STATIC_ASSERT(( dim == 2 ), "Bad Dimension");
      elems[ 0 ] = x;
      elems[ 1 ] = y;
    }

    /**
     * Initialize a 3D vector
     */
    __device__ __host__
    Vector( const T& x, const T& y, const T& z )
    {
      STATIC_ASSERT(( dim == 3 ), "Bad Dimension");
      elems[ 0 ] = x;
      elems[ 1 ] = y;
      elems[ 2 ] = z;
    }

    /**
     * Initialize a 4D vector
     */
    __device__ __host__
    Vector( const T& x, const T& y, const T& z, const T& t )
    {
      STATIC_ASSERT(( dim == 4 ), "Bad Dimension");
      elems[ 0 ] = x;
      elems[ 1 ] = y;
      elems[ 2 ] = z;
      elems[ 3 ] = t;
    }

    /**
     * Returns a raw pointer on the data
     */
    __device__ __host__
    T* data() { return elems; }

    /**
     * STL-iteration begin
     */
    __device__ __host__
    iterator begin() { return elems; }
    /**
     * Stl-iteration constant begin
     */
    __device__ __host__
    const_iterator begin() const { return elems; }

    /**
     * STL-iteration end
     */
    __device__ __host__
    iterator end() { return elems+dim; }
    /**
     * Stl-iteration constant end
     */
    __device__ __host__
    const_iterator end() const { return elems+dim; }

    /**
     * Returns a constant raw pointer on the data
     */
    __device__ __host__
    const T* c_data() const { return elems; }

    /**
     * Vector negation
     */
    __device__ __host__
    Vector operator-(void) const
    {
      Vector ans;
      for(size_t i = 0 ; i < dim ; i++)
        ans[i] = -elems[i];

      return ans;
    }

    /**
     * Vector addition
     */
    __device__ __host__
    Vector operator+(const Vector& vec) const
    {
      Vector ans(*this);
      ans += vec;
      return ans;
    }

    /**
     * Vector subtraction
     */
    __device__ __host__
    Vector operator-(const Vector& vec) const
    {
      Vector ans(*this);
      ans -= vec;
      return ans;
    }

    /**
     * Element-wise multiplcation
     */
    __device__ __host__
    Vector mult(const Vector& vec) const
    {
      Vector ans;
      for(size_t i = 0 ; i < dim ; i++)
        ans[i] = elems[i] * vec.elems[i];
      return ans;
    }

    /**
     * Multiplication by a scalar
     */
    __device__ __host__
    Vector operator*(const T& scalar) const
    {
      Vector ans(*this);
      ans *= scalar;
      return ans;
    }

    /**
     * Division by a scalar
     */
    __device__ __host__
    Vector operator/(const T& scalar) const
    {
      Vector ans(*this);
      ans /= scalar;
      return ans;
    }

    /**
     * Element-wise division
     */
    __device__ __host__
    Vector operator/(const Vector& vec) const
    {
      Vector ans = *this;
      ans /= vec;
      return ans;
    }

    /**
     * In-place element-wise division by a scalar
     */
    __device__ __host__
    Vector& operator/=(const Vector& vec)
    {
      for(size_t i = 0 ; i < dim ; ++i)
        elems[i] /= vec.elems[i];
      return *this;
    }

    /**
     * Multiplication by a scalar
     */
    __device__ __host__
    friend Vector operator*(const T& scalar,const Vector& vec)
    {
      Vector ans = vec;
      ans *= scalar;
      return ans;
    }

//    __device__ __host__
//    friend Vector operator*(const T& scalar,const Vector& vec)
//    {
//      Vector ans;
//      for(size_t i = 0 ; i < dim ; i++)
//        ans[i] = scalar * vec.elems[i];
//
//      return ans;
//    }

    /**
     * Dot product
     */
    __device__ __host__
    T operator*(const Vector& vec) const
    {
      T ans = 0;
      for(size_t i = 0 ; i < dim ; i++)
        ans += elems[i] * vec.elems[i];

      return ans;
    }

    /**
     * Vector copy
     */
    __device__ __host__
    Vector& operator=(const Vector& vec)
    {
      for(size_t i = 0 ; i < dim ; i++)
        elems[i] = vec.elems[i];

      return (*this);
    }

    /**
     * In-place vector addition
     */
    __device__ __host__
    Vector& operator+=(const Vector& vec)
    {
      for(size_t i = 0 ; i < dim ; i++)
        elems[i] += vec.elems[i];
      return *this;
    }

    /**
     * In-place vector subtraction
     */
    __device__ __host__
    Vector& operator-=(const Vector& vec)
    {
      for(size_t i = 0 ; i < dim ; i++)
        elems[i] -= vec.elems[i];
      return *this;
    }

    /**
     * In-place multiplication by a scalar
     */
    __device__ __host__
    //Vector& operator*=(const T& scalar)
    Vector& operator*=(const float scalar)
    {
      for(size_t i = 0 ; i < dim ; i++)
        elems[i] *= scalar;
      return *this;
    }

    /**
     * In-place multiplication by a scalar
     */
//    template <typename T1>
//    __device__ __host__
//    Vector& operator*=(const T1& scalar)
//    {
//      for(size_t i = 0 ; i < dim ; i++)
//        elems[i] = (T)(elems[i]*scalar);
//      return *this;
//    }

    /**
     * In-place division by a scalar
     */
    __device__ __host__
    Vector& operator/=(const T& scalar)
    {
      for(size_t i = 0 ; i < dim ; ++i)
        elems[i] /= scalar;
      return *this;
    }

    /**
     * In-place division by a scalar
     */
    template <typename T1>
    __device__ __host__
    Vector& operator/=(const T1& scalar)
    {
      for(size_t i = 0 ; i < dim ; ++i)
        elems[i] = (T)(elems[i] / scalar);
      return *this;
    }

    /**
     * Element-wise equality
     */
    __device__ __host__
    bool operator==(const Vector& vec) const
    {
      for(size_t i = 0 ; i < dim ; i++)
        if(elems[i] != vec.elems[i])
          return false;

      return true;
    }

    /**
     * Element-wise inequality
     */
    __device__ __host__
    bool operator!=(const Vector& vec) const
    {
      return (!((*this) == vec));
    }

    /**
     * Access to the element \c idx
     */
    __device__ __host__
    T& operator[](size_t idx)
    {
      return elems[idx];
    }

    /**
     * Access to the element \c idx
     */
    __device__ __host__
    T operator[](size_t idx) const
    {
      return elems[idx];
    }

    /**
     * Euclidean norm of the vector
     */
    __device__ __host__
    T norm() const
    {
      return std::sqrt(normsq());
    }

    /**
     * Square of the Euclidean norm of the vector
     */
    __device__ __host__
    T normsq() const
    {
      T ans = 0;
      for(size_t i = 0 ; i < dim ; i++)
        ans += elems[i] * elems[i];

      return ans;
    }

    /**
     * Normalize the vector
     */
    __device__ __host__
    Vector& normalize(void)
    {
      T sz = norm();
      return ((*this) /= sz);
    }

    /**
     * Returns a normalized version of the vector
     */
    __device__ __host__
    Vector normalized(void) const
    {
      Vector ans(*this);
      return ans.normalize();
    }

    __device__ __host__
    bool iszero(void)
    {
      for(size_t i = 0 ; i < dim ; i++)
        if(elems[i] != 0)
          return false;
      return true;
    }

    __device__ __host__
    Vector& zero(void)
    {
      for(size_t i = 0 ; i < dim ; i++)
        elems[i] = 0;
      return (*this);
    }

    /**
     * Set the values of a 1-D vector
     */
    __device__ __host__
    void set( const T& x )
    {
      STATIC_ASSERT(( dim == 1 ), "Bad Dimension");
      elems[ 0 ] = x;
    }

    /**
     * Set the values of a 2-D vector
     */
    __device__ __host__
    void set( const T& x, const T& y )
    {
      STATIC_ASSERT(( dim == 2 ), "Bad Dimension");
      elems[ 0 ] = x;
      elems[ 1 ] = y;
    }

    /**
     * Set the values of a 3-D vector
     */
    __device__ __host__
    void set( const T& x, const T& y, const T& z )
    {
      STATIC_ASSERT(( dim == 3 ), "Bad Dimension");
      elems[ 0 ] = x;
      elems[ 1 ] = y;
      elems[ 2 ] = z;
    }

    /**
     * Set the values of a 4-D vector
     */
    __device__ __host__
    void set( const T& x, const T& y, const T& z, const T& t )
    {
      STATIC_ASSERT(( dim == 4 ), "Bad Dimension");
      elems[ 0 ] = x;
      elems[ 1 ] = y;
      elems[ 2 ] = z;
      elems[ 3 ] = t;
    }

    /**
     * Set all the elements to \c value
     */
    __device__ __host__
    Vector& operator=( const T& value )
    {
      for( size_t i = 0 ; i < dim ; ++i )
      {
        elems[ i ] = value;
      }
      return *this;
    }

    /**
     * Compute the cross product as \c this x \c other
     */
    __device__ __host__
    Vector cross( const Vector& other ) const
    {
      STATIC_ASSERT(( dim == 3 ), "Bad Dimension");
      return ( *this ) ^ other;
    }

    /**
     * Short access to the first element
     */
    __device__ __host__
    void x(const T& v) { STATIC_ASSERT(( dim > 0 ), "Bad Dimension"); elems[0] = v; }
    /**
     * Short access to the second element
     */
    __device__ __host__
    void y(const T& v) { STATIC_ASSERT(( dim > 1 ), "Bad Dimension"); elems[1] = v; }
    /**
     * Short access to the third element
     */
    __device__ __host__
    void z(const T& v) { STATIC_ASSERT(( dim > 2 ), "Bad Dimension"); elems[2] = v; }
    /**
     * Short access to the fourth element
     */
    __device__ __host__
    void t(const T& v) { STATIC_ASSERT(( dim > 3 ), "Bad Dimension"); elems[3] = v; }
    /**
     * Short access to the first element
     */
    __device__ __host__
    T& x() { STATIC_ASSERT(( dim > 0 ), "Bad Dimension"); return elems[0]; }
    /**
     * Short access to the second element
     */
    __device__ __host__
    T& y() { STATIC_ASSERT(( dim > 1 ), "Bad Dimension"); return elems[1]; }
    /**
     * Short access to the third element
     */
    __device__ __host__
    T& z() { STATIC_ASSERT(( dim > 2 ), "Bad Dimension"); return elems[2]; }
    /**
     * Short access to the fourth element
     */
    __device__ __host__
    T& t() { STATIC_ASSERT(( dim > 3 ), "Bad Dimension"); return elems[3]; }
    /**
     * Short access to the first element
     */
    __device__ __host__
    const T& x() const { STATIC_ASSERT(( dim > 0 ), "Bad Dimension"); return elems[0]; }
    /**
     * Short access to the second element
     */
    __device__ __host__
    const T& y() const { STATIC_ASSERT(( dim > 1 ), "Bad Dimension"); return elems[1]; }
    /**
     * Short access to the third element
     */
    __device__ __host__
    const T& z() const { STATIC_ASSERT(( dim > 2 ), "Bad Dimension"); return elems[2]; }
    /**
     * Short access to the fourth element
     */
    __device__ __host__
    const T& t() const { STATIC_ASSERT(( dim > 3 ), "Bad Dimension"); return elems[3]; }

    /**
     * Short access to the first element
     */
    __device__ __host__
    void i(const T& v) { STATIC_ASSERT(( dim > 0 ), "Bad Dimension"); elems[0] = v; }
    /**
     * Short access to the second element
     */
    __device__ __host__
    void j(const T& v) { STATIC_ASSERT(( dim > 1 ), "Bad Dimension"); elems[1] = v; }
    /**
     * Short access to the third element
     */
    __device__ __host__
    void k(const T& v) { STATIC_ASSERT(( dim > 2 ), "Bad Dimension"); elems[2] = v; }
    /**
     * Short access to the fourth element
     */
    __device__ __host__
    void l(const T& v) { STATIC_ASSERT(( dim > 3 ), "Bad Dimension"); elems[3] = v; }
    /**
     * Short access to the first element
     */
    __device__ __host__
    T& i() { STATIC_ASSERT(( dim > 0 ), "Bad Dimension"); return elems[0]; }
    /**
     * Short access to the second element
     */
    __device__ __host__
    T& j() { STATIC_ASSERT(( dim > 1 ), "Bad Dimension"); return elems[1]; }
    /**
     * Short access to the third element
     */
    __device__ __host__
    T& k() { STATIC_ASSERT(( dim > 2 ), "Bad Dimension"); return elems[2]; }
    /**
     * Short access to the fourth element
     */
    __device__ __host__
    T& l() { STATIC_ASSERT(( dim > 3 ), "Bad Dimension"); return elems[3]; }
    /**
     * Short access to the first element
     */
    __device__ __host__
    const T& i() const { STATIC_ASSERT(( dim > 0 ), "Bad Dimension"); return elems[0]; }
    /**
     * Short access to the second element
     */
    __device__ __host__
    const T& j() const { STATIC_ASSERT(( dim > 1 ), "Bad Dimension"); return elems[1]; }
    /**
     * Short access to the third element
     */
    __device__ __host__
    const T& k() const { STATIC_ASSERT(( dim > 2 ), "Bad Dimension"); return elems[2]; }
    /**
     * Short access to the fourth element
     */
    __device__ __host__
    const T& l() const { STATIC_ASSERT(( dim > 3 ), "Bad Dimension"); return elems[3]; }

    /**
     * Extract the two first elements of the vector
     */
    __device__ __host__
    Vector<2,T> projectXY(void)
    {
      STATIC_ASSERT(( dim>1 ), "Bad Dimension");
      return Vector<2,T>(elems[0],elems[1]);
    }

    friend std::ostream& operator<<(std::ostream& out,const Vector& vec)
    {
      for(size_t i = 0 ; i < dim ; i++)
      {
        out << vec.elems[i];
        if(i != (dim - 1))
          out << " ";
      }
      return out;
    }

    friend std::istream& operator>>(std::istream& in,Vector& vec)
    {
      in >> vec[ 0 ];
      for(size_t i = 1 ; i < dim && in ; i++)
        in >> std::ws >> vec[i];
      return in;
    }
/*
    friend QTextStream& operator<<(QTextStream& out, const Vector& vec)
    {
      for(size_t i = 0 ; i < dim ; i++)
      {
        out << vec.elems[i];
        if(i != (dim - 1))
          out << " ";
      }
      return out;
    }

    friend QTextStream& operator>>(QTextStream& in,Vector& vec)
    {
      in >> vec[ 0 ];
      for(size_t i = 1 ; i < dim && !in.atEnd() ; i++)
        in >> ws >> vec[i];
      return in;
    }
*/
  };

  /**
   * Cross product \c v1 x \c v2
   */
  template <class T>
  __device__ __host__
  T operator%( const Vector<2,T>& v1, const Vector<2,T>& v2 )
  {
    return v1^v2;
  }

  /**
   * Cross product \c v1 x \c v2 (French notation)
   */
  template <class T>
  __device__ __host__
  T operator^( const Vector<2,T>& v1, const Vector<2,T>& v2 )
  {
    return ((v1[0] * v2[1]) -
            (v1[1] * v2[0]));
  }

  /**
   * Cross product \c v1 x \c v2 (French notation)
   */
  template <class T>
  __device__ __host__
  T operator^( const Vector<1,T>& v1, const Vector<1,T>& v2 )
  {
    return 0;
  }

  /**
   * Cross product \c v1 x \c v2
   */
  template <class T>
  __device__ __host__
  Vector<3,T> operator%(const Vector<3,T>& v1,const Vector<3,T>& v2)
  {
    return v1^v2;
  }

  /**
   * Cross product \c v1 x \c v2 (French notation)
   */
  template <class T>
  __device__ __host__
  Vector<3,T> operator^(const Vector<3,T>& v1,const Vector<3,T>& v2)
  {
    Vector<3,T> ans;
    ans[0] = v1[1]*v2[2] - v1[2]*v2[1];
    ans[1] = v1[2]*v2[0] - v1[0]*v2[2];
    ans[2] = v1[0]*v2[1] - v1[1]*v2[0];
    return ans;
  }

  /**
   * Angle of the vector with (0,1)
   * \relates Vector
   */
  template <class T>
  __device__ __host__
  double angle( const Vector<2,T>& v )
  {
    return atan2( v.y(), v.x() );
  }

  /**
   * Non-oriented angle between \c v1 and \c v2
   * \relates Vector
   */
  template <class T>
  __device__ __host__
  double angle( const Vector<3,T>& v1, const Vector<3,T>& v2 )
  {
    double x = v1*v2;
    double y = norm( v1^v2 );
    return atan2( y, x );
  }

  /**
   * Oriented angle between \c v1 and \c v2
   * \relates Vector
   */
  template <class T>
  __device__ __host__
  double angle( const Vector<2,T>& v1, const Vector<2,T>& v2 )
  {
    double x = v1*v2;
    double y = v1^v2;
    return atan2( y, x );
  }

  /**
   * Oriented angle between \c v1 and \c v2
   * \relates Vector
   */
  template <class T>
  __device__ __host__
  double angle( const Vector<1,T>& v1, const Vector<1,T>& v2 )
  {
    return ( v1*v2 < 0 )? -1 : 1;
  }

  /**
   * Oriented angle between \c v1 and \c v2 with \c ref to orient the space.
   * \relates Vector
   */
  template <class T>
  __device__ __host__
  double angle( const Vector<3,T>& v1, const Vector<3,T>& v2, const Vector<3,T>& ref )
  {
    double x = v1*v2;
    Vector<3,T> n = v1^v2;
    double y = norm( n );
    if( n*ref < 0 )
      return atan2( -y, x );
    else
      return atan2( y, x );
  }

  /**
   * Euclidian square norm of a real
   *
   * Just the square of the real
   */
  __device__ __host__
  double normalized(double)
  {
    return 1;
  }

  /**
   * Euclidian square norm of a real
   *
   * Just the square of the real
   */
  __device__ __host__
  double normsq(double s)
  {
    return s*s;
  }

  /**
   * Euclidian norm of a real
   *
   * Just the absolute value of that real
   */
  __device__ __host__
  double norm(double s)
  {
    return (s<0)?-s:s;
  }

  /**
   * Function-version of the norm
   *
   * \relates Vector
   * \see Vector::norm()
   */
  template <size_t dim, typename T>
  __device__ __host__
  T norm( const Vector<dim,T>& v )
  {
    return v.norm();
  }

  /**
   * Function-version of the square norm
   *
   * \relates Vector
   * \see Vector::normsq()
   */
  template <size_t dim, typename T>
  __device__ __host__
  T normsq( const Vector<dim,T>& v )
  {
    return v.normsq();
  }

  /**
   * Function-version of the square norm
   *
   * \relates Vector
   * \see Vector::normsq()
   */
  template <size_t dim, typename T>
  __device__ __host__
  Vector<dim,T> normalized( const Vector<dim,T>& v )
  {
    return v.normalized();
  }

  /**
   * Return the vector whose component is the max of the two input vectors 
   * components.
   *
   * \relates Vector
   */
  template <size_t dim, typename T>
  __device__ __host__
  Vector<dim,T> max( const Vector<dim,T>& v1, const Vector<dim,T>& v2)
  {
    Vector<dim,T> result;
    for(int i = 0 ; i < dim ; ++i)
    {
      result[i] = std::max(v1[i], v2[i]);
    }
    return result;
  }

  /**
   * Return the vector whose component is the min of the two input vectors 
   * components.
   *
   * \relates Vector
   */
  template <size_t dim, typename T>
  __device__ __host__
  Vector<dim,T> min( const Vector<dim,T>& v1, const Vector<dim,T>& v2)
  {
    Vector<dim,T> result;
    for(int i = 0 ; i < dim ; ++i)
    {
      result[i] = std::min(v1[i], v2[i]);
    }
    return result;
  }

  /**
   * Find a vector orthogonal to v
   *
   * \relates util::Vector
   */
  template <typename T>
  __device__ __host__
  util::Vector<3,T> orthogonal(const util::Vector<3,T>& v)
  {
    const double ratio = 1-1e-8;
    if ((std::abs(v.y()) >= ratio*std::abs(v.x())) && (std::abs(v.z()) >= ratio*std::abs(v.x())))
      return util::Vector<3,T>(0, -v.z(), v.y());
    else
      if ((std::abs(v.x()) >= ratio*std::abs(v.y())) && (std::abs(v.z()) >= ratio*std::abs(v.y())))
        return util::Vector<3,T>(-v.z(), 0, v.x());
      else
        return util::Vector<3,T>(-v.y(), v.x(), 0);
  }

  template <size_t dim, typename T>
  __device__ __host__
  Vector<dim,T> map(const T& (*fct)(const T&), const Vector<dim,T>& v)
  {
    Vector<dim,T> result;
    for(size_t i = 0 ; i < dim ; ++i)
    {
      result[i] = (*fct)(v[i]);
    }
    return result;
  }

  template <size_t dim, typename T>
  __device__ __host__
  Vector<dim,T> map(T (*fct)(const T&), const Vector<dim,T>& v)
  {
    Vector<dim,T> result;
    for(size_t i = 0 ; i < dim ; ++i)
    {
      result[i] = (*fct)(v[i]);
    }
    return result;
  }

  template <size_t dim, typename T>
  __device__ __host__
  Vector<dim,T> map(T (*fct)(T), const Vector<dim,T>& v)
  {
    Vector<dim,T> result;
    for(size_t i = 0 ; i < dim ; ++i)
    {
      result[i] = (*fct)(v[i]);
    }
    return result;
  }

  template <size_t dim, typename T, typename T1>
  __device__ __host__
  Vector<dim,T> map(const T& (*fct)(const T1&), const Vector<dim,T1>& v)
  {
    Vector<dim,T> result;
    for(size_t i = 0 ; i < dim ; ++i)
    {
      result[i] = (*fct)(v[i]);
    }
    return result;
  }

  template <size_t dim, typename T, typename T1>
  __device__ __host__
  Vector<dim,T> map(T (*fct)(const T1&), const Vector<dim,T1>& v)
  {
    Vector<dim,T> result;
    for(size_t i = 0 ; i < dim ; ++i)
    {
      result[i] = (*fct)(v[i]);
    }
    return result;
  }

  /**
   * Map a function to all elements of a vector
   */
  template <size_t dim, typename T, typename T1>
  __device__ __host__
  Vector<dim,T> map(T (*fct)(T1), const Vector<dim,T1>& v)
  {
    Vector<dim,T> result;
    for(size_t i = 0 ; i < dim ; ++i)
    {
      result[i] = (*fct)(v[i]);
    }
    return result;
  }

  template <size_t dim, typename T>
  __device__ __host__
  Vector<dim,T> map(const T& (*fct)(const T&,const T&), const Vector<dim,T>& v1, const Vector<dim,T>& v2)
  {
    Vector<dim,T> result;
    for(size_t i = 0 ; i < dim ; ++i)
    {
      result[i] = (*fct)(v1[i], v2[i]);
    }
    return result;
  }

  template <size_t dim, typename T>
  __device__ __host__
  Vector<dim,T> map(T (*fct)(const T&,const T&), const Vector<dim,T>& v1, const Vector<dim,T>& v2)
  {
    Vector<dim,T> result;
    for(size_t i = 0 ; i < dim ; ++i)
    {
      result[i] = (*fct)(v1[i], v2[i]);
    }
    return result;
  }

  template <size_t dim, typename T>
  __device__ __host__
  Vector<dim,T> map(T (*fct)(T,T), const Vector<dim,T>& v1, const Vector<dim,T>& v2)
  {
    Vector<dim,T> result;
    for(size_t i = 0 ; i < dim ; ++i)
    {
      result[i] = (*fct)(v1[i], v2[i]);
    }
    return result;
  }

  template <size_t dim, typename T, typename T1, typename T2>
  __device__ __host__
  Vector<dim,T> map(const T& (*fct)(const T1&,const T2&), const Vector<dim,T1>& v1, const Vector<dim,T2>& v2)
  {
    Vector<dim,T> result;
    for(size_t i = 0 ; i < dim ; ++i)
    {
      result[i] = (*fct)(v1[i], v2[i]);
    }
    return result;
  }

  template <size_t dim, typename T, typename T1, typename T2>
  __device__ __host__
  Vector<dim,T> map(T (*fct)(const T1&,const T2&), const Vector<dim,T1>& v1, const Vector<dim,T2>& v2)
  {
    Vector<dim,T> result;
    for(size_t i = 0 ; i < dim ; ++i)
    {
      result[i] = (*fct)(v1[i], v2[i]);
    }
    return result;
  }

  /**
   * Map a function to all elements of a vector
   */
  template <size_t dim, typename T, typename T1, typename T2>
  __device__ __host__
  Vector<dim,T> map(T (*fct)(T1,T2), const Vector<dim,T1>& v1, const Vector<dim,T2>& v2)
  {
    Vector<dim,T> result;
    for(size_t i = 0 ; i < dim ; ++i)
    {
      result[i] = (*fct)(v1[i], v2[i]);
    }
    return result;
  }

}


#endif // __VECTOR_H
