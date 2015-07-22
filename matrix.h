#ifndef _VVLIB_UTIL_MATRIX_H
#define _VVLIB_UTIL_MATRIX_H
/**
 * \file matrix.h
 *
 * Defines the util::Matrix class template
 */

#include <config.h>
#include "vector.h"
#include <util/static_assert.h>
#include <cmath>

namespace util
{
  /**
   * \class Matrix matrix.h <util/matrix.h>
   *
   * Class representing a fixed-size matrix.
   *
   * This class is optimized for small-sized matrix (3x3 or 4x4).
   */
  template <size_t nRows,size_t nCols,typename T = double>
    class Matrix
    {
    private:
      Vector<nCols,T> rows[nRows];

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
      Matrix() {}

      /**
       * Copy a matrix.
       *
       * Can be used to cast a matrix onto another type.
       */
      template <typename T1>
        __device__ __host__
        Matrix(const Matrix<nRows,nCols,T1>& mat)
        {
          for(size_t i = 0 ; i < nRows ; i++)
            rows[i] = mat.rows[i];
        }

      /**
       * Fill the matrix with the array of vectors.
       *
       * \param vecs Array of \c nRows vectors.
       */
      template <typename T1>
        __device__ __host__
        Matrix(const Vector<nCols,T1>* vecs)
        {
          for(size_t i = 0 ; i < nRows ; i++)
            rows[i] = vecs[i];
        }

      /**
       * Fill in the matrix with the \c values.
       *
       * \param values nRows*nCols array. If \c c_style is true, \c values is 
       * rows first (i.e. the first values correspond to the first row). 
       * Otherwise, values are columns first.
       *
       * \param c_style Determine the ordering of values.
       */
      __device__ __host__
      Matrix( const T* values, bool c_style = true)
      {
        if(c_style)
        {
          for(size_t i = 0 ; i < nRows ; i++)
          {
            rows[i] = Vector<nCols,T>(values + (i*nCols));
          }
        }
        else
        {
          for(size_t i = 0 ; i < nRows ; i++)
            for(size_t j = 0 ; j < nCols ; j++)
            {
              rows[i][j] = values[i+j*nRows];
            }
        }
      }

      /**
       * Create a diagonal matrix.
       *
       * \param value Value placed on the diagonal.
       */
      __device__ __host__
      Matrix(const T& value)
      {
        for(size_t i = 0 ; i < nRows ; i++)
          for(size_t j = 0 ; j < nCols ; j++)
            rows[i][j] = (i==j)?value:0;
      }

      /**
       * Returns a constant raw pointer on the data
       */
      __device__ __host__
      const T* c_data() const { return rows[0].c_data(); }

      /**
       * Matrix subtraction
       */
      __device__ __host__
      Matrix operator-(void) const
      {
        Matrix ans;

        for(size_t i = 0 ; i < nRows ; i++)
          ans.rows[i] = -rows[i];

        return ans;
      }

      /**
       * Matrix addition
       */
      __device__ __host__
      Matrix operator+(const Matrix& mat) const
      {
        Matrix ans;

        for(size_t i = 0 ; i < nRows ; i++)
          ans.rows[i] = rows[i] + mat.rows[i];

        /**
         * Matrix subtraction
         */
        return ans;
      }

      __device__ __host__
      Matrix operator-(const Matrix& mat) const
      {
        Matrix ans;

        for(size_t i = 0 ; i < nRows ; i++)
          ans.rows[i] = rows[i] - mat.rows[i];

        return ans;
      }

      /**
       * Matrix-scalar multiplication
       */
      __device__ __host__
      Matrix operator*(const T& scalar) const
      {
        Matrix ans;

        for(size_t i = 0 ; i < nRows ; i++)
          ans[i] = rows[i] * scalar;

        return ans;
      }

      /**
       * Matrix-scalar division
       */
      __device__ __host__
      Matrix operator/(const T& scalar) const
      {
        Matrix ans;

        for(size_t i = 0 ; i < nRows ; i++)
          ans[i] = rows[i] / scalar;

        return ans;
      }

      /**
       * Matrix-scalar multiplication
       */
      __device__ __host__
      friend Matrix operator*(const T& scalar,const Matrix& mat)
      {
        Matrix ans;

        for(size_t i = 0 ; i < nRows ; i++)
          ans[i] = scalar * mat.rows[i];

        return ans;
      }

      /**
       * Matrix*Vector
       */
      __device__ __host__
      Vector<nRows,T> operator*(const Vector<nRows,T>& vec) const
      {
        Vector<nRows, T> ans;
        for(size_t i = 0; i < nRows; i++) {
	  ans[i] = 0;
          for(size_t j = 0; j < nCols; j++)
            ans[i] += rows[i][j] * vec[j];
	}
        return ans;
      }

/*
      __device__ __host__
      Vector<nRows,T> operator*(const Vector<nRows,T>& vec) const
      {
        Matrix<nRows,1,T> mat;
        for(size_t i = 0; i < nRows; i++)
          mat[i][0] = vec[i];
        mat = (*this) * mat;
        Vector<nRows, T> ans;
        for(size_t i = 0; i < nRows; i++)
          ans[i] = mat[i][0];

        return ans;
      }
*/

      __device__ __host__
      Matrix& operator=(const Matrix& mat)
      {
        for(size_t i = 0 ; i < nRows ; i++)
          rows[i] = mat.rows[i];

        return (*this);
      }

      __device__ __host__
      Matrix& operator+=(const Matrix& mat)
      {
        return ((*this) = (*this) + mat);
      }

      __device__ __host__
      Matrix& operator-=(const Matrix& mat)
      {
        return ((*this) = (*this) - mat);
      }

      __device__ __host__
      Matrix& operator*=(const T& scalar)
      {
        return ((*this) = (*this) * scalar);
      }

      __device__ __host__
      Matrix& operator/=(const T& scalar)
      {
        return ((*this) = (*this) / scalar);
      }

      __device__ __host__
      Matrix& operator*=(const Matrix& mat)
      {
        return ((*this) = (*this) * mat);
      }

      __device__ __host__
      bool operator==(const Matrix& mat) const
      {
        for(size_t i = 0 ; i < nRows ; i++)
          if(rows[i] != mat.rows[i])
            return false;

        return true;
      }

      __device__ __host__
      bool operator!=(const Matrix& mat) const
      {
        return (!((*this) == mat));
      }

      friend std::ostream& operator<<(std::ostream& out,const Matrix& mat)
      {
        for(size_t i = 0 ; i < nRows ; i++)
        {
          out << mat.rows[i];
          if(i != (nRows - 1))
            out << " ";
        }

        return out;
      }

      friend std::istream& operator>>(std::istream& in,Matrix& mat)
      {
        for(size_t i = 0 ; i < nRows && in ; i++)
          in >> mat.rows[i];
        return in ;
      }

      /**
       * Returns the nth row
       *
       * \param idx Index of the returned row
       */
      __device__ __host__
      Vector<nCols,T>& operator[](size_t idx)
      {
        return rows[idx];
      }

      /**
       * Returns the nth row
       *
       * \param idx Index of the returned row
       */
      //__device__ __host__
      //Vector<nCols,T> operator[](size_t idx) const
      //{
      // return rows[idx];
      //}

      /**
       * Return the value at row \c i, column \c j.
       */
      __device__ __host__
      T& operator()(size_t i,size_t j)
      {
        return rows[i][j];
      }

      /**
       * Return the value at row \c i, column \c j.
       */
      __device__ __host__
      T operator()(size_t i,size_t j) const
      {
        return rows[i][j];
      }

      /**
       * Returns an identity matrix.
       */
      __device__ __host__
      static Matrix identity()
      {
        Matrix mat(1);
        return mat;
      }

      /**
       * Set the matrix to all zero.
       */
      __device__ __host__
      Matrix& zero(void)
      {
        for(size_t i = 0 ; i < nRows ; i++)
          for(size_t j = 0 ; j < nCols ; j++)
            rows[i][j] = 0.0;
        return (*this);
      }

      /**
       * Set the matrix to a diagonal matrix.
       *
       * \param value Value to put on the diagonal
       */
      __device__ __host__
      Matrix& operator=( const T& value )
      {
        for( size_t i = 0 ; i < nRows ; ++i )
        {
          for( size_t j = 0 ; j < nCols ; ++j )
          {
            if( i == j )
              rows[ i ][ j ] = value;
            else
              rows[ i ][ j ] = 0;
          }
        }
        return *this;
      }

      /**
       * Transpose the matrix
       */
      __device__ __host__
      Matrix<nCols,nRows,T> operator~()
      {
        Matrix<nCols,nRows,T> t;
        for( size_t i = 0 ; i < nRows ; ++i )
          for( size_t j = 0 ; j < nCols ; ++j )
            t[ i ][ j ] = rows[ j ][ i ];
        return t;
      }

      /**
       * Creates the 3x3 matrix corresponding to a rotation.
       *
       * \param direction Axes of the rotation
       * \param angle Angle of the rotation
       */
      __device__ __host__
      static Matrix<3,3,T> rotation( const Vector<3, T>& direction, T angle )
      {
        T ca = std::cos( angle );
        T sa = std::sin( angle );
        Matrix<3,3,T> r;
        double x = direction.x();
        double y = direction.y();
        double z = direction.z();
        r[ 0 ].set( ca+(1-ca)*x*x,   (1-ca)*x*y-sa*z, (1-ca)*z*x+sa*y );
        r[ 1 ].set( (1-ca)*y*x+sa*z, ca+(1-ca)*y*y,   (1-ca)*z*y-sa*x );
        r[ 2 ].set( (1-ca)*x*z-sa*y, (1-ca)*y*z+sa*x, ca+(1-ca)*z*z );
        return r;
      }

      /**
       * Creates the 4x4 matrix corresponding to a rotation.
       *
       * \param direction Axes of the rotation
       * \param angle Angle of the rotation
       */
      __device__ __host__
      static Matrix<4,4,T> rotation( const Vector<4, T>& direction, T angle )
      {
        T ca = std::cos( angle );
        T sa = std::sin( angle );
        Matrix<4,4,T> r;
        double x = direction.x()/direction.t();
        double y = direction.y()/direction.t();
        double z = direction.z()/direction.t();
        r[ 0 ].set( ca+(1-ca)*x*x,   (1-ca)*x*y-sa*z, (1-ca)*z*x+sa*y, 0 );
        r[ 1 ].set( (1-ca)*y*x+sa*z, ca+(1-ca)*y*y,   (1-ca)*z*y-sa*x, 0 );
        r[ 2 ].set( (1-ca)*x*z-sa*y, (1-ca)*y*z+sa*x, ca+(1-ca)*z*z, 0 );
        r[ 3 ].set( 0, 0, 0, 1 );
        return r;
      }

      /**
       * Trace of the matrix
       */
      __device__ __host__
      T trace() const
      {
        T acc = 0;
        for(size_t i = 0 ; i < std::min(nRows,nCols) ; ++i)
        {
          acc += rows[i][i];
        }
        return acc;
      }

      void fillArray(T* array, bool row_first = true)
      {
        if(row_first)
        {
          memcpy(array, &rows[0][0], sizeof(T)*nRows*nCols);
        }
        else
        {
          size_t k = 0;
          for(size_t c = 0 ; c < nCols ; ++c)
            for(size_t r = 0 ; r < nRows ; ++r, ++k)
              array[k] = rows[r][c];
        }
      }

    };

  /**
   * Matrix multiplication
   */
  template<size_t nRows,size_t nSize,size_t nCols,typename T>
    __device__ __host__
    Matrix<nRows,nCols,T> operator*(const Matrix<nRows,nSize,T>& mat1,
                                    const Matrix<nSize,nCols,T>& mat2);
  /**
   * Determinant of the matrix
   */
  template<typename T> 
    __device__ __host__
    T det(const Matrix<1,1,T>& mat);
  /**
   * Determinant of the matrix
   */
  template<typename T>
    __device__ __host__
    T det(const Matrix<2,2,T>& mat);
  /**
   * Determinant of the matrix
   */
  template<typename T> 
    __device__ __host__
    T det(const Matrix<3,3,T>& mat);
  /**
   * Determinant of the matrix
   *
   * \warning the method used is \f$O(n^3)\f$ complexity !
   */
  template <size_t nRows, typename T> 
    __device__ __host__
    T det(const Matrix<nRows,nRows,T>& mat);

  template <size_t nRows, typename T>
    __device__ __host__
    T matrix_minor(const Matrix<nRows,nRows,T>& mat, size_t i, size_t j);

  /**
   * Returns the cofactor of the matrix for position (i,j)
   */
  template <size_t nRows, typename T>
    __device__ __host__
    T cofactor(const Matrix<nRows,nRows,T>& mat, size_t i, size_t j);

  /**
   * Inverse the matrix
   *
   * \relates Matrix
   */
  template <typename T>
    __device__ __host__
    Matrix<1,1,T> inverse(const Matrix<1,1,T>& mat);
  /**
   * Inverse the matrix
   *
   * \relates Matrix
   */
  template <typename T>
    __device__ __host__
    Matrix<2,2,T> inverse(const Matrix<2,2,T>& mat);
  /**
   * Inverse the matrix
   *
   * \relates Matrix
   */
  template <typename T>
    __device__ __host__
    Matrix<3,3,T> inverse(const Matrix<3,3,T>& mat);

  /**
   * Inverse the matrix
   *
   * \relates Matrix
   *
   * \warning This algorithm is sub-optimal
   */
  template <size_t nRows, typename T>
    __device__ __host__
    Matrix<nRows,nRows,T> inverse(const Matrix<nRows,nRows,T>& mat);

  /**
   * Transpose a matrix.
   *
   * \relates Matrix
   */
  template <size_t nRows, size_t nCols, typename T>
    __device__ __host__
    Matrix<nCols,nRows,T> transpose( const Matrix<nRows, nCols, T>& mat);

  /**
   * Return the norm of the matrix.
   *
   * The norm is defined as the square-root of the sum of the square of the 
   * values.
   *
   * \relates Matrix
   */
  template <size_t nRows, size_t nCols, typename T>
    __device__ __host__
    T norm(const Matrix<nRows,nCols,T>& mat);

  /**
   * Return the square norm of the matrix.
   *
   * \see norm(const Matrix&)
   *
   * \relates Matrix
   */
  template <size_t nRows, size_t nCols, typename T>
    __device__ __host__
    T normsq(const Matrix<nRows,nCols,T>& mat);

  template <size_t nRows, size_t nCols, typename T>
  __device__ __host__
  Matrix<nRows,nCols,T> map(const T& (*fct)(const T&), const Matrix<nRows,nCols,T>& m)
  {
    Matrix<nRows,nCols,T> result;
    for(size_t i = 0 ; i < nRows ; ++i)
    {
      const Vector<nCols,T>& mrow = m[i];
      Vector<nCols,T>& rrow = result[i];
      for(size_t j = 0 ; j < nCols ; ++j)
      {
        rrow[j] = (*fct)(mrow[j]);
      }
    }
    return result;
  }

  template <size_t nRows, size_t nCols, typename T>
  __device__ __host__
  Matrix<nRows,nCols,T> map(T (*fct)(T), const Matrix<nRows,nCols,T>& m)
  {
    Matrix<nRows,nCols,T> result;
    for(size_t i = 0 ; i < nRows ; ++i)
    {
      const Vector<nCols,T>& mrow = m[i];
      Vector<nCols,T>& rrow = result[i];
      for(size_t j = 0 ; j < nCols ; ++j)
      {
        rrow[j] = (*fct)(mrow[j]);
      }
    }
    return result;
  }


  template <size_t nRows, size_t nCols, typename T>
  __device__ __host__
  Matrix<nRows,nCols,T> map(T (*fct)(const T&), const Matrix<nRows,nCols,T>& m)
  {
    Matrix<nRows,nCols,T> result;
    for(size_t i = 0 ; i < nRows ; ++i)
    {
      const Vector<nCols,T>& mrow = m[i];
      Vector<nCols,T>& rrow = result[i];
      for(size_t j = 0 ; j < nCols ; ++j)
      {
        rrow[j] = (*fct)(mrow[j]);
      }
    }
    return result;
  }

  template <size_t nRows, size_t nCols, typename T, typename T1>
  __device__ __host__
  Matrix<nRows,nCols,T> map(const T& (*fct)(const T1&), const Matrix<nRows,nCols,T1>& m)
  {
    Matrix<nRows,nCols,T> result;
    for(size_t i = 0 ; i < nRows ; ++i)
    {
      const Vector<nCols,T1>& mrow = m[i];
      Vector<nCols,T>& rrow = result[i];
      for(size_t j = 0 ; j < nCols ; ++j)
      {
        rrow[j] = (*fct)(mrow[j]);
      }
    }
    return result;
  }

  template <size_t nRows, size_t nCols, typename T, typename T1>
  __device__ __host__
  Matrix<nRows,nCols,T> map(T (*fct)(T1), const Matrix<nRows,nCols,T1>& m)
  {
    Matrix<nRows,nCols,T> result;
    for(size_t i = 0 ; i < nRows ; ++i)
    {
      const Vector<nCols,T1>& mrow = m[i];
      Vector<nCols,T>& rrow = result[i];
      for(size_t j = 0 ; j < nCols ; ++j)
      {
        rrow[j] = (*fct)(mrow[j]);
      }
    }
    return result;
  }


  template <size_t nRows, size_t nCols, typename T, typename T1>
  __device__ __host__
  Matrix<nRows,nCols,T> map(T (*fct)(const T1&), const Matrix<nRows,nCols,T1>& m)
  {
    Matrix<nRows,nCols,T> result;
    for(size_t i = 0 ; i < nRows ; ++i)
    {
      const Vector<nCols,T1>& mrow = m[i];
      Vector<nCols,T>& rrow = result[i];
      for(size_t j = 0 ; j < nCols ; ++j)
      {
        rrow[j] = (*fct)(mrow[j]);
      }
    }
    return result;
  }

  template <size_t nRows, size_t nCols, typename T>
  __device__ __host__
  Matrix<nRows,nCols,T> map(T (*fct)(T,T), const Matrix<nRows,nCols,T>& m1, const Matrix<nRows,nCols,T>& m2)
  {
    Matrix<nRows,nCols,T> result;
    for(size_t i = 0 ; i < nRows ; ++i)
    {
      const Vector<nCols,T>& mrow1 = m1[i];
      const Vector<nCols,T>& mrow2 = m2[i];
      Vector<nCols,T>& rrow = result[i];
      for(size_t j = 0 ; j < nCols ; ++j)
      {
        rrow[j] = (*fct)(mrow1[j], mrow2[j]);
      }
    }
    return result;
  }

  template <size_t nRows, size_t nCols, typename T>
  __device__ __host__
  Matrix<nRows,nCols,T> map(T (*fct)(const T&, const T&), const Matrix<nRows,nCols,T>& m1, const Matrix<nRows,nCols,T>& m2)
  {
    Matrix<nRows,nCols,T> result;
    for(size_t i = 0 ; i < nRows ; ++i)
    {
      const Vector<nCols,T>& mrow1 = m1[i];
      const Vector<nCols,T>& mrow2 = m2[i];
      Vector<nCols,T>& rrow = result[i];
      for(size_t j = 0 ; j < nCols ; ++j)
      {
        rrow[j] = (*fct)(mrow1[j], mrow2[j]);
      }
    }
    return result;
  }

  template <size_t nRows, size_t nCols, typename T>
  __device__ __host__
  Matrix<nRows,nCols,T> map(const T& (*fct)(const T&, const T&), const Matrix<nRows,nCols,T>& m1, const Matrix<nRows,nCols,T>& m2)
  {
    Matrix<nRows,nCols,T> result;
    for(size_t i = 0 ; i < nRows ; ++i)
    {
      const Vector<nCols,T>& mrow1 = m1[i];
      const Vector<nCols,T>& mrow2 = m2[i];
      Vector<nCols,T>& rrow = result[i];
      for(size_t j = 0 ; j < nCols ; ++j)
      {
        rrow[j] = (*fct)(mrow1[j], mrow2[j]);
      }
    }
    return result;
  }

  template <size_t nRows, size_t nCols, typename T, typename T1, typename T2>
  __device__ __host__
  Matrix<nRows,nCols,T> map(T (*fct)(T1,T2), const Matrix<nRows,nCols,T1>& m1, const Matrix<nRows,nCols,T2>& m2)
  {
    Matrix<nRows,nCols,T> result;
    for(size_t i = 0 ; i < nRows ; ++i)
    {
      const Vector<nCols,T1>& mrow1 = m1[i];
      const Vector<nCols,T2>& mrow2 = m2[i];
      Vector<nCols,T>& rrow = result[i];
      for(size_t j = 0 ; j < nCols ; ++j)
      {
        rrow[j] = (*fct)(mrow1[j], mrow2[j]);
      }
    }
    return result;
  }

  template <size_t nRows, size_t nCols, typename T, typename T1, typename T2>
  __device__ __host__
  Matrix<nRows,nCols,T> map(T (*fct)(const T1&, const T2&), const Matrix<nRows,nCols,T1>& m1, const Matrix<nRows,nCols,T2>& m2)
  {
    Matrix<nRows,nCols,T> result;
    for(size_t i = 0 ; i < nRows ; ++i)
    {
      const Vector<nCols,T1>& mrow1 = m1[i];
      const Vector<nCols,T2>& mrow2 = m2[i];
      Vector<nCols,T>& rrow = result[i];
      for(size_t j = 0 ; j < nCols ; ++j)
      {
        rrow[j] = (*fct)(mrow1[j], mrow2[j]);
      }
    }
    return result;
  }

  template <size_t nRows, size_t nCols, typename T, typename T1, typename T2>
  __device__ __host__
  Matrix<nRows,nCols,T> map(const T& (*fct)(const T1&, const T2&), const Matrix<nRows,nCols,T1>& m1, const Matrix<nRows,nCols,T2>& m2)
  {
    Matrix<nRows,nCols,T> result;
    for(size_t i = 0 ; i < nRows ; ++i)
    {
      const Vector<nCols,T1>& mrow1 = m1[i];
      const Vector<nCols,T2>& mrow2 = m2[i];
      Vector<nCols,T>& rrow = result[i];
      for(size_t j = 0 ; j < nCols ; ++j)
      {
        rrow[j] = (*fct)(mrow1[j], mrow2[j]);
      }
    }
    return result;
  }

#include <util/matrix_impl.h>

}

#endif // __MATRIX_H
