#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/iterator/counting_iterator.h>
#include "vector.h"
#include "matrix.h"


#include <cuda_runtime.h>
#include <cublas.h>

typedef util::Vector<3, float> Vec3f;

#define MPREC float
//#define MPREC double
typedef thrust::host_vector<float> HVecF;
typedef thrust::host_vector<unsigned int> HVecU;
typedef thrust::host_vector<unsigned char> HVecB;
typedef thrust::device_vector<float> DVecF;
typedef thrust::device_vector<unsigned int> DVecU;
typedef thrust::device_vector<unsigned char> DVecB;

using std::cout;
using std::cerr;
using std::endl;

// Move Point based on forces
struct movePointsOP
{
  Vec3f *pos, *vel, *force, *box;
  MPREC damp, friction, threshold, dt;

  movePointsOP(Vec3f *_pos, Vec3f *_vel, Vec3f *_force, Vec3f *_box, MPREC _damp, MPREC _friction, MPREC _threshold, MPREC _dt) : 
    pos(_pos), vel(_vel), force(_force), box(_box), damp(_damp), friction(_friction), threshold(_threshold), dt(_dt) {}

  __device__
    void operator()(const int vtx) const 
    { 
      vel[vtx] += force[vtx] / damp;
      pos[vtx] += dt * vel[vtx];

      for(int i = 0; i < 3; i ++) {
        if(pos[vtx][i] > box[0][i]) {
          pos[vtx] -= vel[vtx] * dt;
          pos[vtx][i] = box[0][i];
          vel[vtx][i] = 0;
          pos[vtx] += dt * vel[vtx];
        }
        if(pos[vtx][i] < -box[0][i]) {
          pos[vtx] -= dt * vel[vtx];
          pos[vtx][i] = -box[0][i];
          vel[vtx][i] = 0;
          pos[vtx] += dt * vel[vtx];
        }
      }
      return;
    }
};

// Find forces
struct findForcesOP
{
  Vec3f *pos, *force;
  float *restlen, kr, dt;
  unsigned int *nbhd, nbs;


  findForcesOP(unsigned int *_nbhd, Vec3f *_pos, Vec3f *_force, float *_restlen, Vec3f *_gravity, float _kr, float _dt, unsigned int _nbs) : 
    nbhd(_nbhd), pos(_pos), force(_force), restlen(_restlen), kr(_kr), dt(_dt), nbs(_nbs) {}

  __device__
    void operator()(const int vtx) const 
    { 
      unsigned int edg = vtx * nbs;
      Vec3f dir;

      for(unsigned int i = 0; i < nbs; i++, edg++)
        if(nbhd[edg] != vtx) {
          dir = pos[nbhd[edg]] - pos[vtx];
          float len = norm(dir);
          dir /= len;
          force[vtx] += dir * (len - restlen[edg]) * kr;
        }

      return;
    }
};

// Error checking
cudaError_t checkCudaError(char *msg)
{
  cudaError_t cuerr = cudaGetLastError();
  if(cuerr != cudaSuccess)
    fprintf(stderr, "CUDA Error %d: %s\n", cuerr, msg);
  return(cuerr);
}

static const unsigned int maxn = 100;

// Distributed objects
static DVecU* Nhbd[maxn];  // Neighborhood information
struct NInfo {
  unsigned int n;    // Number of vertices in graph 
  unsigned int nbs;    // Max number of neighbors
} NInfo[maxn];
static DVecB* VData[maxn];  // Vertex data
static DVecB* EData[maxn];  // Edge data

extern "C" {

  int initDNC()
  {
    static bool initialized = false;
    if(!initialized) {
      for(int i = 0; i < maxn; i++) {
        VData[i] = 0; EData[i] = 0; Nhbd[i] = 0; 
        NInfo[i].n = NInfo[i].nbs = 0;
      }
      initialized = true;

      // Initialize cuda
      int count, i;
      cudaGetDeviceCount(&count);
      if(count == 0) {
        cerr << "There is no cuda device." << endl;
        return(1);
      }

      for(i = 0; i < count; i++) {
        cudaDeviceProp prop;
        if(cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
          if(prop.major >= 1) {
            break;
          }
        }
      }
      if(i == count) {
        cerr << "There is no device supporting CUDA." << endl;
        return(2);
      }
      cudaSetDevice(i);
      checkCudaError("Setting cuda device");
    }
    return(0);
  }

  int allocDeviceDNC(unsigned int idx, unsigned int n, unsigned int nbs)
  {
    if(idx > maxn)
      return(1);

    if(!Nhbd[idx] || NInfo[idx].n != n || NInfo[idx].nbs != nbs) {
      if(Nhbd[idx])
        delete(Nhbd[idx]);
      Nhbd[idx] = 0;
      if(n * nbs > 0)
        Nhbd[idx] = new DVecU(n * nbs);
      NInfo[idx].n = n;
      NInfo[idx].nbs = nbs;
    }
    if(!Nhbd[idx]) {
      cout << "allocDeviceDNC::Error allocating buffer" << endl;
      return(2);
    }

    return(0);
  }

  int allocDeviceDVC(unsigned int nidx, unsigned int idx, unsigned int sz)
  {
    if(idx > maxn)
      return(1);

    if(!VData[idx] || (VData[idx])->size() != NInfo[nidx].n * sz) {
      if(VData[idx])
        delete(VData[idx]);
      VData[idx] = 0;
      if(NInfo[nidx].n * sz > 0)
        VData[idx] = new DVecB(NInfo[nidx].n * sz);
    }
    if(!VData[idx]) {
      cout << "allocDeviceDVC::Error allocating buffer" << endl;
      return(2);
    }

    return(0);
  }

  int allocDeviceDEC(unsigned int nidx, unsigned int idx, unsigned int sz)
  {
    if(idx > maxn)
      return(1);

    if(!EData[idx] || (EData[idx])->size() != NInfo[nidx].n * NInfo[nidx].nbs * sz) {
      if(EData[idx])
        delete(EData[idx]);
      EData[idx] = 0;
      if(NInfo[nidx].n * NInfo[nidx].nbs * sz > 0)
        EData[idx] = new DVecB(NInfo[nidx].n * NInfo[nidx].nbs * sz);
    }
    if(!EData[idx]) {
      cout << "allocDeviceDNC::Error allocating buffer" << endl;
      return(2);
    }

    return(0);
  }

  int copyToDeviceDNC(unsigned int r, HVecU &nhbd)
  {
    if(r > maxn || !Nhbd[r] || nhbd.size() != (Nhbd[r])->size()) {
      cout << "copyToDeviceDNC::Error r:" << r << " size:" << nhbd.size() << " alloc size:" << (Nhbd[r])->size() << endl;
      return(1);
    }

    *Nhbd[r] = nhbd;
    return(0);
  }

  int copyToDeviceDVC(unsigned int r, HVecB &vdata)
  {
    if(r > maxn || !VData[r] || vdata.size() != (VData[r])->size()) {
      cout << "copyToDeviceDVC::Error r:" << r << " size:" << vdata.size() << " alloc size:" << (VData[r])->size() << endl;
      return(1);
    }
    *VData[r] = vdata;
    return(0);
  }

  int copyToHostDVC(unsigned int r, HVecB &vdata)
  {
    if(r > maxn || !VData[r] || vdata.size() != (VData[r])->size()) {
      cout << "copyToHostDVC::Error r:" << r << " size:" << vdata.size() << " alloc size:" << (VData[r])->size() << endl;
      return(1);
    }
    vdata = *VData[r];
    return(0);
  }

  int copyToDeviceDEC(unsigned int r, HVecB &edata)
  {
    if(r > maxn || !EData[r] || edata.size() != (EData[r])->size()) {
      cout << "copyToDeviceDEC::Error r:" << r << " size:" << edata.size() << " alloc size:" << (EData[r])->size() << endl;
      return(1);
    }
    *EData[r] = edata;
    return(0);
  }

  int massSpring(unsigned int nb, unsigned int pos, unsigned int vel, unsigned int force, unsigned int restlen, 
      float *box, float kr, float damp, float friction, float threshold, float dt, unsigned int steps)
  {
    if(nb > maxn || !Nhbd[nb] || pos > maxn || !VData[pos] || vel > maxn || !VData[vel] || force > maxn || !VData[force]) 
      return(1);

    DVecF TBox(box, box + 3);
    Vec3f *Box = (Vec3f *)(&TBox[0]).get();
    Vec3f *Pos = (Vec3f *)(&(*VData[pos])[0]).get();
    Vec3f *Vel = (Vec3f *)(&(*VData[vel])[0]).get();
    Vec3f *Force = (Vec3f *)(&(*VData[force])[0]).get();
    MPREC *Restlen = (MPREC *)(&(*EData[restlen])[0]).get();
    unsigned int *Nb = (unsigned int *)(&(*Nhbd[nb])[0]).get();
/*
    thrust::counting_iterator<int, thrust::device_space_tag> first(0);
    thrust::counting_iterator<int, thrust::device_space_tag> last(NInfo[nb].n);

    findForcesOP ffOP(Nb, Pos, Force, Restlen, Gravity, kr, dt, NInfo[nb].nbs);
    movePointsOP mpOP(Pos, Vel, Force, Box, damp, friction, threshold, dt);

    for(unsigned int i = 0; i < steps; i++) {
      thrust::for_each(first, last, ffOP);
      cudaThreadSynchronize();
      thrust::for_each(first, last, mpOP);
      cudaThreadSynchronize();
    }*/
    return(0);
  }
}
