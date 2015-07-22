#include "thrust/host_vector.h"

#define byte char

typedef thrust::host_vector<uint> HVecU;
typedef thrust::host_vector<byte> HVecB;

extern "C" {
  int initDNC();
  int allocDeviceDNC(uint nidx, uint n, uint nbs);
  int copyToDeviceDNC(uint nidx, HVecU &nhbd);
  int allocDeviceDVC(uint nidx, uint idx, uint sz);
  int copyToDeviceDVC(uint idx, HVecB &vdata);
  int copyToHostDVC(uint idx, HVecB &vdata);
  int allocDeviceDEC(uint nidx, uint idx, uint sz);
  int copyToDeviceDEC(uint idx,  HVecB &edata);
}

// Distributed neighborhood object
template <typename vvgraph>
class DNObj {
  public:
    // Type of a vertex
    typedef typename vvgraph::vertex_t vertex;
    // Type of an edge
    typedef typename vvgraph::edge_t edge;

    // Default constructor
    DNObj() : n(0), nbs(0) {}

    // Set offset of data item in vertex
    DNObj(vvgraph *p_S) : S(p_S)
  {
    initIdx();
  }

    // Return graph
    vvgraph& graph(){
      return S;
    };

    // Allocate and send to GPU
    void allocCopyToDevice()
    {
      // Record vertex numbers and find max # of neighbors
      vnum.clear();
      n = nbs = 0;
      forall(const vertex &v, *S)  {
        vnum[v] = n++;
        if(S->valence(v) > nbs)
          nbs = S->valence(v);
      }
      HVecU nbdata(n * nbs);
      uint d = 0;
      // Grab the neighborhood information 
      forall(const vertex &v, *S) {
        // Put in edge data, 
        forall(const vertex &w, S->neighbors(v))
          nbdata[d++] = vnum[w];

        // Use current index to indicate empty
        for(uint k = S->valence(v); k < nbs; k++)
          nbdata[d++] = vnum[v];
      }
      int err = allocDeviceDNC(idx, n, nbs);
      if(err)
        std::cout << "Error allocating neighborhood data on device:" << err << " idx:" << idx << std::endl;
      err = copyToDeviceDNC(idx, nbdata);
      if(err)
        std::cout << "Error copying neighborhood data to device:" << err << " idx:" << idx << std::endl;
    }

    //friend class DVObj<vvgraph, typename VDataType>;
    //friend class DEObj<vvgraph, typename EDataType>;
    //friend class DVObj;
    //friend class DEObj;

    uint idx;
    vvgraph *S;
    uint n;
    uint nbs;
    std::map<vertex, uint> vnum;

  private:
    void initIdx()
    {
      static uint nextIdx = 0;
      if(nextIdx == 0)
        initDNC();
      idx = nextIdx++;
    }
};

// Distributed vertex object
template <typename vvgraph, typename DNObj, typename VDataType>
class DVObj {
  public:
    // Type of a vertex
    typedef typename vvgraph::vertex_t vertex;
    // Type of an edge
    typedef typename vvgraph::edge_t edge;

    // Default constructor
    DVObj() {}

    // Set offset of data item in vertex
    DVObj(DNObj *p_nhbd, size_t p_off) : nhbd(p_nhbd), off(p_off)
  {
    initIdx();
  }

    // Get data 
    inline byte *get(const vertex &v)
    {
      return (byte *)&*v + off;
    }

    // Allocate storage for a distributed vertex on GPU
    void allocDevice()
    {
      if(nhbd->n <= 0)
        return;
      int err = allocDeviceDVC(nhbd->idx, idx, sizeof(VDataType));
      if(err)
        std::cout << "Error allocating vertex data on device:" << err << " idx:" << idx << std::endl;
    }

    // Copy vertex data to GPU
    void copyToDevice()
    {
      if(nhbd->n <= 0)
        return;
      HVecB vdata(nhbd->n * sizeof(VDataType));
      forall(const vertex &v, *nhbd->S) {
        uint d = nhbd->vnum[v] * sizeof(VDataType);
        byte *s = get(v);
        for(uint j = 0; j < sizeof(VDataType); j++)
          vdata[d++] = *s++;
      }
      int err = copyToDeviceDVC(idx, vdata);
      if(err)
        std::cout << "Error copying vertex data to device:" << err << " idx:" << idx << std::endl;
    }

    // Copy vertex data to Host
    void copyToHost()
    {
      if(nhbd->n <= 0)
        return;
      HVecB vdata(nhbd->n * sizeof(VDataType));
      int err = copyToHostDVC(idx, vdata);
      if(err)
        std::cout << "Error copying vertex data to host:" << err << " idx:" << idx << std::endl;
      forall(const vertex &v, *nhbd->S) {
        uint s = nhbd->vnum[v] * sizeof(VDataType);
        byte *d = get(v);
        for(uint j = 0; j < sizeof(VDataType); j++)
          *d++ = vdata[s++];
      }
    }

    uint idx;
  private:
    DNObj *nhbd;
    size_t off;

    void initIdx()
    {
      static uint nextIdx = 0;
      idx = nextIdx++;
    }
};

// Distributed edge object
template <typename vvgraph, typename DNObj, typename EDataType>
class DEObj {
  public:
    // Type of a vertex
    typedef typename vvgraph::vertex_t vertex;
    // Type of an edge
    typedef typename vvgraph::edge_t edge;

    // Default constructor
    DEObj() {}

    // Set offset of data item in edge
    DEObj(DNObj *p_nhbd, size_t p_off) : nhbd(p_nhbd), off(p_off)
  {
    initIdx();
  }

    // Get edge data
    inline byte *get(const edge &e)
    {
      return (byte *)&*e + off;
    }

    // Allocate storage for a distributed edge on GPU
    void allocDevice()
    {
      if(nhbd->n * nhbd->nbs <= 0)
        return;
      int err = allocDeviceDEC(nhbd->idx, idx, sizeof(EDataType));
      if(err)
        std::cout << "Error allocating edge data on device:" << err << " idx:" << idx << std::endl;
    }

    // Send edge data to GPU
    void copyToDevice()
    {
      if(nhbd->n * nhbd->nbs <= 0)
        return;
      HVecB edata(nhbd->n * nhbd->nbs * sizeof(EDataType));
      // Grab data
      uint d = 0;
      forall(const vertex &v, *nhbd->S) {
        //uint d = nhbd->vnum[v] * nhbd->nbs * sizeof(EDataType);
        forall(const vertex &w, nhbd->S->neighbors(v)) {
          byte *s = get(nhbd->S->edge(v,w));
          for(uint j = 0; j < sizeof(EDataType); j++)
            edata[d++] = *s++;
        }
        // Pad with 0s
        for(uint k = nhbd->S->valence(v); k < nhbd->nbs; k++)
          for(uint j = 0; j < sizeof(EDataType); j++)
            edata[d++] = 0;
      }
      int err = copyToDeviceDEC(idx, edata);
      if(err)
        std::cout << "Error copying edge data to device:" << err << " idx:" << idx << std::endl;
    }

    // Copy edge data to host
    void copyToHost()
    {
      if(nhbd->n * nhbd->nbs <= 0)
        return;
    }

    uint idx;
  private:
    DNObj *nhbd;
    size_t off;

    void initIdx()
    {
      static uint nextIdx = 0;
      idx = nextIdx++;
    }
};
