#include <vve.h>

#include <string>
#include <math.h>

#include <geometry/geometry.h>
#include <algorithms/insert.h>
#include <util/parms.h>
#include <util/palette.h>
#include <util/materials.h>
#include <iostream>
#include <iomanip>
#include <iterator> 
#include <map>

#include "dlobject.h"

#include <QFile>

using std::cout;
using std::endl;
using std::string;

typedef util::Vector<3, int> Vec3i;
typedef util::Vector<3, size_t> Vec3size;
typedef util::Vector<3, float> Vec2f;
typedef util::Vector<3, float> Vec3f;
using algorithms::insert;

extern "C" int massSpring(uint nhbd, uint pos, uint vel, uint force, uint restlen, const float *box, const float *gravity,
                                  float kr, float damp, float friction, float threshold, float dt, unsigned int frameSteps);

class VertexData
{
public:
  // Vertex model data
  Vec3f pos;					// postition
  Vec3f vel;					// velocity
  Vec3f force;					// Uniform force on points

  Vec3f RK4 [8];                                 //Data for Runge-Khutta method

  int n;					// number of filament vertex belongs to
  bool barb;					// barbed or pointed end
  bool draw;					// Draw vertex

  // Constructor, set initial values
  VertexData() : pos(0,0,0), vel(0,0,0), force(0,0,0), barb(false), draw(false) {
                    RK4[0]=Vec3f(0,0,0);
                    RK4[1]=Vec3f(0,0,0);
                    RK4[2]=Vec3f(0,0,0);
                    RK4[3]=Vec3f(0,0,0);
                    RK4[4]=Vec3f(0,0,0);
                    RK4[5]=Vec3f(0,0,0);
                    RK4[6]=Vec3f(0,0,0);
                    RK4[7]=Vec3f(0,0,0);
                    }
};

class EdgeData
{
public:
  float restlen;				// rest length of springs
  bool draw;

  // Constructor, set initial values
  EdgeData(): restlen(0), draw(false) {}
};

class FilaminData
{
public:
  Vec3f pos;
  Vec3f vel;
  Vec3f force;

  Vec3f RK4 [4]; 

  bool draw;
  
  VVGraph<VertexData, EdgeData>::vertex_t* v[4];
  FilaminData():  pos(0,0,0), vel(0,0,0), force(0,0,0), draw(true) {
	      v[0]=NULL;
	      v[1]=NULL;
	      v[2]=NULL;
	      v[3]=NULL;
	      RK4[0]=Vec3f(0,0,0);
	      RK4[1]=Vec3f(0,0,0);
	      RK4[2]=Vec3f(0,0,0);
	      RK4[3]=Vec3f(0,0,0);
	      }
};

// Type of the VV graph
typedef VVGraph<VertexData, EdgeData> vvgraph;
typedef VVGraph<FilaminData, EdgeData> fgraph;

// Type of a vertex
typedef vvgraph::vertex_t vertex;
typedef fgraph::vertex_t filamin;


// Type of an edge
typedef vvgraph::edge_t edge;

// Type of distributed object neighborhood object
typedef DNObj<vvgraph> DNObjT;


class HashData
{
public:
  std::list<vertex*> vlist;
  std::list<filamin*> flist;
  HashData(): vlist(), flist() {}
};

// Class definition of model
class myModel : public Model
{
public:
  // Parameters
  bool Debug;
  float Dt;					// timingstep
  Vec3f BoxSize;				// Size of box
  float CellSize;				// Size of cells
  bool WireFrame;				// Draw wireframe or shaded
  float DrawNormals;				// Size to draw normals
  float DrawForce;				// Size to force vectors
  bool DrawBox;					// Draw the box
  float FrameSteps;				// Steps per frame
  bool PrintStats;				// Print out stats
  bool Parallel;				// Use Cuda
  int SeedRandom;        			// Seed for random numbers
  int TimeSolver;
  int N;						
  bool showhashinfo;

  float kStretchActin;				
  float kBendActin;
  float DampA;					// Spring dampening
  bool PrintForce;				// Print out force vectors
  float Gravity;				// Gravity strength
  float Friction;				// Friction when cube hits wall
  float Threshold;				// Threshold velocity under which vertices on surface don't move
  float RestLenght;
  float VarianceA;
  int Nfil;
  int Ngr;

  float kStretchFilamin;
  float kBendFilamin;
  float RestLenghtFilamin;
  float DampF;
  float VarianceF;

private:
  // Model variables
  vvgraph S;					// S = Actin Network 
  fgraph F;
  std::vector<vertex> E;                        // Exterior vertices
  std::list<HashData> hash;
  util::Palette palette;			// Colors
  util::Materials materials;			// Materials

  int steps;					// Step count
  float timing;					// timing counter
  Vec3f gravity;

  DNObjT nhbd;					// Distributed objects
  DVObj<vvgraph, DNObjT, Vec3f> pos;
  DVObj<vvgraph, DNObjT, Vec3f> vel;
  DVObj<vvgraph, DNObjT, Vec3f> force;
  DEObj<vvgraph, DNObjT, float> restlen;


public:
  void readParms()
  {
    util::Parms parms("view.v");

    parms("Main", "Debug", Debug);
    parms("Main", "Dt", Dt);
    parms("Main", "CellSize", CellSize);
    parms("Main", "BoxSize", BoxSize);
    parms("Main", "WireFrame", WireFrame);
    parms("Main", "DrawNormals", DrawNormals);
    parms("Main", "DrawForce", DrawForce);
    parms("Main", "DrawBox", DrawBox);
    parms("Main", "FrameSteps", FrameSteps);
    parms("Main", "PrintStats", PrintStats);
    parms("Main", "Parallel", Parallel);
    parms("Main", "SeedRandom", SeedRandom);
    parms("Main", "TimeSolver", TimeSolver);
    parms("Main", "HashSize", N);
    parms("Main", "ShowHashInfo", showhashinfo);

    parms("Actin", "kStretchActin", kStretchActin);
    parms("Actin", "kBendActin", kBendActin);
    parms("Actin", "Damp", DampA);
    parms("Actin", "PrintForce", PrintForce);
    parms("Actin", "Gravity", Gravity);
    parms("Actin", "Friction", Friction);
    parms("Actin", "Thresold", Threshold);
    parms("Actin", "RestLenght", RestLenght);
    parms("Actin", "Variance", VarianceA);
    parms("Actin", "Number of filaments", Nfil);
    parms("Actin", "Number for growth", Ngr);

    parms("Filamin", "kStretchFilamin", kStretchFilamin);
    parms("Filamin", "kBendFilamin", kBendFilamin);
    parms("Filamin", "RestLenghtFilamin", RestLenghtFilamin);
    parms("Filamin", "Damp", DampF);
    parms("Filamin", "Variance", VarianceF);

  }

  // Here, reread the files when they are modified
  void modifiedFiles(const std::set<std::string>& filenames)
  {
    forall(const std::string& file, filenames)
    {
      if(file == "pal.map")
        palette.reread();
      else if(file == "view.v")
        readParms();
    }
  }

  myModel(QObject *parent): Model(parent), palette("pal.map"), materials("material.mat"), nhbd(&S), pos(&nhbd, offsetof(VertexData, pos)),
              vel(&nhbd, offsetof(VertexData, vel)), force(&nhbd, offsetof(VertexData, force)), restlen(&nhbd, offsetof(EdgeData, restlen))
  {
    // Read the parameters
    readParms();
    srand(SeedRandom==0?time(0):SeedRandom);

    Vec3f r,r1,r2;
    for(int i=0;i<Nfil;i++) // SPECIFY NUMBER OF Filaments
      {
        r1=Vec3f(unifRand(.0,  BoxSize[0]),unifRand(.0, BoxSize[1]),unifRand(.0, BoxSize[2])) * CellSize;
        r2=Vec3f(unifRand(r1.x(), RestLenght),unifRand(r1.y(), RestLenght),unifRand(r1.z(), RestLenght));
        //r1 = Vec3f(0.,1.,0.);
        //r2 = Vec3f(-1.,-1.,0.);
        createFilament(r1,r2,RestLenght,i+1);
      }
    //r = Vec3f(unifRand(.0, 2.) * BoxSize[0],unifRand(.0, 2.) * BoxSize[1],unifRand(.0, 2.) * BoxSize[2]) * CellSize;
    //r = Vec3f(1.,-1.,0.);

    for(int i=0;i<Ngr;i++){
        growFilamentRandom_WLC(RestLenght);
    }
    
    createHash();
    updateHash();
    createFilaminsRandom(RestLenght*2);
    updateHash();
    cout.precision(15);

    timing = 0;
    steps = 0;

    nhbd.allocCopyToDevice();
    pos.allocDevice();
    pos.copyToDevice();
    vel.allocDevice();
    force.allocDevice();
    restlen.allocDevice();
    restlen.copyToDevice();
  }

  void step()
  {
    if(Parallel) {
      // Call cuda to do the work
      massSpring(nhbd.idx, pos.idx, vel.idx, force.idx, restlen.idx, BoxSize.c_data(), gravity.c_data(), kStretchActin, DampA, Friction, Threshold, Dt, FrameSteps);
      // Get data back from cuda
      pos.copyToHost();
      // Update timing
      steps += FrameSteps;
      timing += Dt * FrameSteps;

    } else {
      for(int i = 0; i < FrameSteps; i++) {
        if(TimeSolver==0) {
          findForces();
	  movePoints();
          } else if (TimeSolver==1) RungeKutta4_WLC();
	  updateHash();
      }
    if(PrintStats && !(steps % 100))
      cout << "Explicit step. Steps: " << steps << ". timing: " << timing << endl;
    }
  }

  void findForces_WLC()
  {
    forall(const vertex &v, S) v->force=Vec3f(0,0,0);
    forall(const filamin &f, F) f->force=Vec3f(0,0,0);
    vertex n1;
    Vec3f a,b,c,d;
    forall(const vertex &v, S)
    {
      n1 = S.anyIn(v);
      forall(const vertex &n2, S.neighbors(v)){
              if(n2!=n1){
                c = S.target(S.edge(v,n1))->pos;
                d = S.source(S.edge(v,n1))->pos;
                a = normalizedSafe(c-d);
                c = S.target(S.edge(v,n2))->pos;
                d = S.source(S.edge(v,n2))->pos;
                b = normalizedSafe(c-d);
                n1->force += (a ^ (a ^ b)) * kBendActin;
                n2->force += ((a ^ b) ^ b) * kBendActin;
              }
              a = n2->pos;
              b = v->pos;
              v->force += normalizedSafe(a - b) * (norm(a - b) - S.edge(v, n2)->restlen) * kStretchActin;
      }
    }

    forall(const filamin &f, F)
    {
      //bending 0-1
      c = f->pos;
      d = (*(f->v[0]))->pos;
      a = normalizedSafe(c-d);
      d = (*(f->v[1]))->pos;
      b = normalizedSafe(c-d);
      (*(f->v[0]))->force += (a ^ (a ^ b)) * kBendFilamin;
      (*(f->v[1]))->force += ((a ^ b) ^ b) * kBendFilamin;

      //stretching
      a = (*(f->v[0]))->pos;
      b = f->pos;
      f->force += normalizedSafe(a - b) * (norm(a - b) - RestLenghtFilamin) * kStretchFilamin;
      (*(f->v[0]))->force += normalizedSafe(b - a) * (norm(a - b) - RestLenghtFilamin) * kStretchFilamin;

      a = (*(f->v[1]))->pos;
      b = f->pos;
      f->force += normalizedSafe(a - b) * (norm(a - b) - RestLenghtFilamin) * kStretchFilamin;
      (*(f->v[1]))->force += normalizedSafe(b - a) * (norm(a - b) - RestLenghtFilamin) * kStretchFilamin;
      
      //shoulder 0-2
      c = f->pos;
      d = (*(f->v[0]))->pos;
      a = normalizedSafe(c-d);
      c = (*(f->v[2]))->pos;
      b = normalizedSafe(c-d);
      f->force +=  abs(a*b) * normalizedSafe((a ^ b) ^ a) * kBendFilamin;
      (*(f->v[2]))->force += abs(a*b) * normalizedSafe(b ^ (a ^ b)) * kBendFilamin;

      //shoulder 1-3
      c = f->pos;
      d = (*(f->v[1]))->pos;
      a = normalizedSafe(c-d);
      c = (*(f->v[3]))->pos;
      b = normalizedSafe(c-d);
      f->force +=  abs(a*b) * normalizedSafe((a ^ b) ^ a) * kBendFilamin;
      (*(f->v[3]))->force += abs(a*b) * normalizedSafe(b ^ (a ^ b)) * kBendFilamin;

    }
  }
  
  void RungeKutta4_WLC()
  {
    vertex n1;
    Vec3f a,b,c,d;

    forall(const vertex &v, S)  for(int i=0;i<8;i++) v->RK4[i]=Vec3f(0,0,0);
    forall(const filamin &f, F) for(int i=0;i<4;i++) f->RK4[i]=Vec3f(0,0,0);

    findForces_WLC();
    //k0
    forall(const vertex &v, S){
      v->RK4[0] = Dt * v->force / DampA;
    }
    forall(const filamin &f, F){
      f->RK4[0] = Dt * f->force / DampF;
    }

    //k1
    forall(const vertex &v, S){
      n1 = S.anyIn(v);
      forall(const vertex &n2, S.neighbors(v)){
              if(n2!=n1){
                c = S.target(S.edge(v,n1))->pos + .5*S.target(S.edge(v,n1))->RK4[0];
                d = S.source(S.edge(v,n1))->pos + .5*S.source(S.edge(v,n1))->RK4[0];
                a = normalizedSafe(c-d);
                c = S.target(S.edge(v,n2))->pos + .5*S.target(S.edge(v,n2))->RK4[0];
                d = S.source(S.edge(v,n2))->pos + .5*S.source(S.edge(v,n2))->RK4[0];
                b = normalizedSafe(c-d);
                // here ^ is cross product
                n1->RK4[1] += (a ^ (a ^ b)) * kBendActin * Dt / DampA;
                n2->RK4[1] += ((a ^ b) ^ b) * kBendActin * Dt / DampA;
              }
              a = n2->pos + .5*n2->RK4[0];
              b = v->pos + .5*v->RK4[0];
              v->RK4[1] += normalizedSafe(a - b) * (norm(a - b) - S.edge(v, n2)->restlen) * kStretchActin * Dt / DampA;
      }
    }
    forall(const filamin &f, F)
    {
      //bending 0-1
      c = f->pos + .5*f->RK4[0];
      d = (*(f->v[0]))->pos + .5*(*(f->v[0]))->RK4[0];
      a = normalizedSafe(c-d);
      d = (*(f->v[1]))->pos + .5*(*(f->v[1]))->RK4[0];
      b = normalizedSafe(c-d);
      (*(f->v[0]))->RK4[1] += (a ^ (a ^ b)) * kBendFilamin * Dt / DampA;
      (*(f->v[1]))->RK4[1] += ((a ^ b) ^ b) * kBendFilamin * Dt / DampA;

      //stretching
      a = (*(f->v[0]))->pos + .5*(*(f->v[0]))->RK4[0];
      b = f->pos + .5*f->RK4[0];
      f->RK4[1] += normalizedSafe(a - b) * (norm(a - b) - RestLenghtFilamin) * kStretchFilamin * Dt / DampF;
      (*(f->v[0]))->RK4[1] += normalizedSafe(b - a) * (norm(a - b) - RestLenghtFilamin) * kStretchFilamin * Dt / DampA;

      a = (*(f->v[1]))->pos + .5*(*(f->v[1]))->RK4[0];
      b = f->pos + .5*f->RK4[0];
      f->RK4[1] += normalizedSafe(a - b) * (norm(a - b) - RestLenghtFilamin) * kStretchFilamin * Dt / DampF;
      (*(f->v[1]))->RK4[1] += normalizedSafe(b - a) * (norm(a - b) - RestLenghtFilamin) * kStretchFilamin * Dt / DampA;

      //shoulder 0-2
      c = f->pos + .5*f->RK4[0];
      d = (*(f->v[0]))->pos + .5*(*(f->v[0]))->RK4[0];
      a = normalizedSafe(c-d);
      c = (*(f->v[2]))->pos + .5*(*(f->v[2]))->RK4[0];
      b = normalizedSafe(c-d);
      f->RK4[1] +=  abs(a*b) * normalizedSafe((a ^ b) ^ a) * kBendFilamin * Dt / DampF;
      (*(f->v[2]))->RK4[1] += abs(a*b) * normalizedSafe(b ^ (a ^ b)) * kBendFilamin * Dt / DampA;

      //shoulder 1-3
      c = f->pos + .5*f->RK4[0];
      d = (*(f->v[1]))->pos + .5*(*(f->v[1]))->RK4[0];
      a = normalizedSafe(c-d);
      c = (*(f->v[3]))->pos + .5*(*(f->v[3]))->RK4[0];
      b = normalizedSafe(c-d);
      f->RK4[1] +=  abs(a*b) * normalizedSafe((a ^ b) ^ a) * kBendFilamin * Dt / DampF;
      (*(f->v[3]))->RK4[1] += abs(a*b) * normalizedSafe(b ^ (a ^ b)) * kBendFilamin * Dt / DampA;
    }
    //k2
    forall(const vertex &v, S){
      n1 = S.anyIn(v);
      forall(const vertex &n2, S.neighbors(v)){
        if(n2!=n1){
            c = S.target(S.edge(v,n1))->pos + .5*S.target(S.edge(v,n1))->RK4[1];
            d = S.source(S.edge(v,n1))->pos + .5*S.source(S.edge(v,n1))->RK4[1];
            a = normalizedSafe(c-d);
            c = S.target(S.edge(v,n2))->pos + .5*S.target(S.edge(v,n2))->RK4[1];
            d = S.source(S.edge(v,n2))->pos + .5*S.source(S.edge(v,n2))->RK4[1];
            b = normalizedSafe(c-d);
            // here ^ is cross product
            n1->RK4[2] += (a ^ (a ^ b)) * kBendActin * Dt / DampA;
            n2->RK4[2] += ((a ^ b) ^ b) * kBendActin * Dt / DampA;
          }
        a = n2->pos + .5*n2->RK4[1];
        b = v->pos + .5*v->RK4[1];
        v->RK4[2] += normalizedSafe(a - b) * (norm(a - b) - S.edge(v, n2)->restlen) * kStretchActin * Dt / DampA;
      }
    }
    forall(const filamin &f, F)
    {
      //bending 0-1
      c = f->pos + .5*f->RK4[1];
      d = (*(f->v[0]))->pos + .5*(*(f->v[0]))->RK4[1];
      a = normalizedSafe(c-d);
      d = (*(f->v[1]))->pos + .5*(*(f->v[1]))->RK4[1];
      b = normalizedSafe(c-d);
      (*(f->v[0]))->RK4[2] += (a ^ (a ^ b)) * kBendFilamin * Dt / DampA;
      (*(f->v[1]))->RK4[2] += ((a ^ b) ^ b) * kBendFilamin * Dt / DampA;

      //stretching
      a = (*(f->v[0]))->pos + .5*(*(f->v[0]))->RK4[1];
      b = f->pos + .5*f->RK4[1];
      f->RK4[2] += normalizedSafe(a - b) * (norm(a - b) - RestLenghtFilamin) * kStretchFilamin * Dt / DampF;
      (*(f->v[0]))->RK4[2] += normalizedSafe(b - a) * (norm(a - b) - RestLenghtFilamin) * kStretchFilamin * Dt / DampA;

      a = (*(f->v[1]))->pos + .5*(*(f->v[1]))->RK4[1];
      b = f->pos + .5*f->RK4[1];
      f->RK4[2] += normalizedSafe(a - b) * (norm(a - b) - RestLenghtFilamin) * kStretchFilamin * Dt / DampF;
      (*(f->v[1]))->RK4[2] += normalizedSafe(b - a) * (norm(a - b) - RestLenghtFilamin) * kStretchFilamin * Dt / DampA;

      //shoulder 0-2
      c = f->pos + .5*f->RK4[1];
      d = (*(f->v[0]))->pos + .5*(*(f->v[0]))->RK4[1];
      a = normalizedSafe(c-d);
      c = (*(f->v[2]))->pos + .5*(*(f->v[2]))->RK4[1];
      b = normalizedSafe(c-d);
      f->RK4[2] +=  abs(a*b) * normalizedSafe((a ^ b) ^ a) * kBendFilamin * Dt / DampF;
      (*(f->v[2]))->RK4[2] += abs(a*b) * normalizedSafe(b ^ (a ^ b)) * kBendFilamin * Dt / DampA;

      //shoulder 1-3
      c = f->pos + .5*f->RK4[1];
      d = (*(f->v[1]))->pos + .5*(*(f->v[1]))->RK4[1];
      a = normalizedSafe(c-d);
      c = (*(f->v[3]))->pos + .5*(*(f->v[3]))->RK4[1];
      b = normalizedSafe(c-d);
      f->RK4[2] +=  abs(a*b) * normalizedSafe((a ^ b) ^ a) * kBendFilamin * Dt / DampF;
      (*(f->v[3]))->RK4[2] += abs(a*b) * normalizedSafe(b ^ (a ^ b)) * kBendFilamin * Dt / DampA;
    }
   //k3
    forall(const vertex &v, S){
      n1 = S.anyIn(v);
      forall(const vertex &n2, S.neighbors(v)){
            if(n2!=n1){
            c = S.target(S.edge(v,n1))->pos + S.target(S.edge(v,n1))->RK4[2];
            d = S.source(S.edge(v,n1))->pos + S.source(S.edge(v,n1))->RK4[2];
            a = normalizedSafe(c-d);
            c = S.target(S.edge(v,n2))->pos + S.target(S.edge(v,n2))->RK4[2];
            d = S.source(S.edge(v,n2))->pos + S.source(S.edge(v,n2))->RK4[2];
            b = normalizedSafe(c-d);
            // here ^ is cross product
            n1->RK4[3] += (a ^ (a ^ b)) * kBendActin * Dt / DampA;
            n2->RK4[3] += ((a ^ b) ^ b) * kBendActin * Dt / DampA;
          }
        a = n2->pos + n2->RK4[2];
        b = v->pos + v->RK4[2];
        v->RK4[3] += normalizedSafe(a - b) * (norm(a - b) - S.edge(v, n2)->restlen) * kStretchActin * Dt / DampA;
      }
    }
    forall(const filamin &f, F)
    {
      //bending 0-1
      c = f->pos + f->RK4[2];
      d = (*(f->v[0]))->pos + (*(f->v[0]))->RK4[2];
      a = normalizedSafe(c-d);
      d = (*(f->v[1]))->pos + (*(f->v[1]))->RK4[2];
      b = normalizedSafe(c-d);
      (*(f->v[0]))->RK4[3] += (a ^ (a ^ b)) * kBendFilamin * Dt / DampA;
      (*(f->v[1]))->RK4[3] += ((a ^ b) ^ b) * kBendFilamin * Dt / DampA;

      //stretching
      a = (*(f->v[0]))->pos + (*(f->v[0]))->RK4[2];
      b = f->pos + f->RK4[2];
      f->RK4[3] += normalizedSafe(a - b) * (norm(a - b) - RestLenghtFilamin) * kStretchFilamin * Dt / DampF;
      (*(f->v[0]))->RK4[3] += normalizedSafe(b - a) * (norm(a - b) - RestLenghtFilamin) * kStretchFilamin * Dt / DampA;

      a = (*(f->v[1]))->pos + (*(f->v[1]))->RK4[2];
      b = f->pos + f->RK4[2];
      f->RK4[3] += normalizedSafe(a - b) * (norm(a - b) - RestLenghtFilamin) * kStretchFilamin * Dt / DampF;
      (*(f->v[1]))->RK4[3] += normalizedSafe(b - a) * (norm(a - b) - RestLenghtFilamin) * kStretchFilamin * Dt / DampA;

      //shoulder 0-2
      c = f->pos + f->RK4[2];
      d = (*(f->v[0]))->pos + (*(f->v[0]))->RK4[2];
      a = normalizedSafe(c-d);
      c = (*(f->v[2]))->pos + (*(f->v[2]))->RK4[2];
      b = normalizedSafe(c-d);
      f->RK4[3] +=  abs(a*b) * normalizedSafe((a ^ b) ^ a) * kBendFilamin * Dt / DampF;
      (*(f->v[2]))->RK4[3] += abs(a*b) * normalizedSafe(b ^ (a ^ b)) * kBendFilamin * Dt / DampA;

      //shoulder 1-3
      c = f->pos + f->RK4[2];
      d = (*(f->v[1]))->pos + (*(f->v[1]))->RK4[2];
      a = normalizedSafe(c-d);
      c = (*(f->v[3]))->pos + (*(f->v[3]))->RK4[2];
      b = normalizedSafe(c-d);
      f->RK4[3] +=  abs(a*b) * normalizedSafe((a ^ b) ^ a) * kBendFilamin * Dt / DampF;
      (*(f->v[3]))->RK4[3] += abs(a*b) * normalizedSafe(b ^ (a ^ b)) * kBendFilamin * Dt / DampA;
    }

  //final step of Runge-Kutta
    forall(const vertex &v, S){
      v->pos += (v->RK4[0]+2*v->RK4[1]+2*v->RK4[2]+v->RK4[3])/6;
      addStochasticForce<vertex>(v,DampA,VarianceA);
      constraints<vertex>(v);
    }
    forall(const filamin &f, F){
      f->pos += (f->RK4[0]+2*f->RK4[1]+2*f->RK4[2]+f->RK4[3])/6;
      addStochasticForce<filamin>(f,DampF,VarianceF);
      constraints<filamin>(f);
    }
      steps++;
      timing += Dt;
  }

  float angleSafe( Vec3f v1, Vec3f v2 )
  {
    float x = v1*v2;
    float y = norm( v1^v2 );
        if(y==0.) return 0.;
        else if(x==0.)  return M_PI/2;
                        else if(isnan(atan2( y, x ))) return 0.;
                              else return atan2( y, x );
  }

  Vec3f normalizedSafe(Vec3f a)
  {
    float tmp = a.x()*a.x()+a.y()*a.y()+a.z()*a.z();
    if(tmp<0) tmp = 0;
    if(tmp == 0) return Vec3f(0.,0.,0.);
    else return(a=a/sqrt(tmp));
  }

  void movePoints()
  {
    forall(const vertex &v, S)
    {
      v->vel += Dt * (v->force - (DampA * v->vel));
      v->pos += Dt * v->vel;
      constraints(v);
      steps++;
      timing += Dt;
    }
  }

  void createFilament(Vec3f r1, Vec3f r2, float restlen, int i)
  {
    vertex v;
    S.insert(v);
    v->pos.x() = r1.x();
    v->pos.y() = r1.y();
    v->pos.z() = r1.z();
    v->draw = true;
    v->barb = true;
    v->n = i;
    vertex w;
    S.insert(w);
    w->pos.x() = r2.x();
    w->pos.y() = r2.y();
    w->pos.z() = r2.z();
    w->draw = true;
    w->n = i;
    insertSpring(v,w,restlen);
  }

  void createFilaminsRandom(float restlen)
  {
    int boo = 0;
    std::list<vertex*> tmp;
    std::list<HashData>::iterator itb;
    for(int i=0;i<N*N*N;i++) {
    itb = hash.begin();
    std::advance(itb,i);
    tmp = itb->vlist;
      //cout << *tmp.begin() << " " << *tmp.end() << endl;
    boo = 0;
    for(std::list<vertex*>::iterator it1 = tmp.begin(); ((it1 != tmp.end()) && boo<10); ++it1){
      //cout << *it1 << endl;
      for(std::list<vertex*>::iterator it2 = it1; ((it2 != tmp.end()) && boo<10); ++it2)
	{
	  if((norm((**it1)->pos-(**it2)->pos) < restlen) && ((**it1)->n != (**it2)->n) && (it1 != it2)) {
	    filamin f;
	    F.insert(f);
	    f->v[0]=*it1;
	    f->v[1]=*it2;
	    f->v[2]=(vertex*)&S.iAnyIn(*(f->v[0]));
	    f->v[3]=(vertex*)&S.iAnyIn(*(f->v[1]));
	    f->pos = ((*(f->v[0]))->pos + (*(f->v[1]))->pos)/2;
	    //cout << i << " " << boo << " " << (*(f->v[0]))->pos << " " << (*(f->v[1]))->pos << " " << endl;
	    boo++;
	    }
	}; 
      };
    };
  }


  void growFilamentRandom_WLC(float restlen)
  {
    bool growthrate = false;
    vertex n = S[rand() % S.size()];
    vertex tmp = n;
    forall(const vertex &w, S.neighbors(n))
      if(tmp != w && S.valence(w) == 1 && growthrate == false) {
        vertex v;
        S.insert(v);
	v->n = w->n;
	Vec3f a = normalizedSafe(w->pos - n->pos);
	Vec3f x = normalizedSafe(Vec3f(a.y(),-a.x(),0.));
	Vec3f y = a ^ x;
	float phi = unifRand(0.,M_PI);
	float r1 = restlen*unifRand(1.,0.01);
	float r2 = (r1 > restlen) ? sqrt(r1*r1-restlen*restlen) : sqrt(restlen*restlen-r1*r1);
	v->pos = w->pos + a*r1 + x*r2*cos(phi) + y*r2*sin(phi);

        v->draw = true;
        insertSpring(v,w,restlen);
        growthrate = true;
      } else if(growthrate < 1) {
        if(tmp != w) tmp=n;
         n=w;
      };
  }

  void insertSpring(vertex v, vertex w, float restlen)
  {
    S.insertEdge(v, w);
    S.edge(v, w)->draw = true;
    S.edge(v, w)->restlen = restlen;
    S.insertEdge(w, v);
    S.edge(w, v)->draw = true;
    S.edge(w, v)->restlen = restlen;
  }




  float unifRand(float mean, float width)
  {
    return(mean + (float(rand())/float(RAND_MAX) - 0.5) * width);
  }
  template <typename T>
  void constraints(const T &v)
  {
    if(v->pos.x() > 0.99*BoxSize.x()) {
            v->pos.x() = BoxSize.x();
	    // v->vel -= Dt * v->vel * Friction * v->vel.x() * v->vel.x();
	    // v->vel = Vec3f(0,0,0);
	    v->vel.x() = 0;
          }
    if(v->pos.x() < -0.99*BoxSize.x()) {
            v->pos.x() = -BoxSize.x();
            //v->vel -= Dt * v->vel * Friction * v->vel.x() * v->vel.x();
            //v->vel = Vec3f(0,0,0);
	    v->vel.x() = 0;
          }
    if(v->pos.y() > 0.99*BoxSize.y()) {
            v->pos.y() = BoxSize.y();
            //v->vel -= Dt * v->vel * Friction * v->vel.y() * v->vel.y();
            //v->vel = Vec3f(0,0,0);
	    v->vel.y() = 0;
          }
    if(v->pos.y() < -0.99*BoxSize.y()) {
            v->pos.y() = -BoxSize.y();
           // v->vel -= Dt * v->vel * Friction * v->vel.y() * v->vel.y();
           // v->vel = Vec3f(0,0,0);
	    v->vel.y() = 0;
          }
    if(v->pos.z() > 0.99*BoxSize.z()) {
            v->pos.z() = BoxSize.z();
            //v->vel -= Dt * v->vel * Friction * v->vel.z() * v->vel.z();
            //v->vel = Vec3f(0,0,0);
	    v->vel.z() = 0;
          }
    if(v->pos.z() < -0.99*BoxSize.z()) {
            v->pos.z() = -BoxSize.z();
            //v->vel -= Dt * v->vel * Friction * v->vel.z() * v->vel.z();
            //v->vel = Vec3f(0,0,0);
	    v->vel.z() = 0;
          }
  }


  //Stochasticity
  template <typename T>
  void addStochasticForce(const T &v, float damp, float variance)
  {
    Vec3f tmp = Vec3f(generateGaussianNoise(.0,variance),generateGaussianNoise(.0,variance),generateGaussianNoise(.0,variance));
    v->pos+=sqrt(Dt)*tmp/damp;
    //cout << tmp.x() << " " << tmp.y() << " " << tmp.z() << " " << endl;
  }

  float generateGaussianNoise(float mu, float  sigma)
{
	const float  epsilon = std::numeric_limits<float >::min();
	const float  tau = 2*M_PI;

	static float  z0, z1;
	static bool generate;
	generate = !generate;

	if (!generate)
	    return z1 * sigma + mu;

	float  u1, u2;
	do{
	    u1 = rand() * (1.0 / RAND_MAX);
	    u2 = rand() * (1.0 / RAND_MAX);
	}
	while ( u1 <= epsilon );

	z0 = sqrt(-2.0 * log(u1)) * cos(tau * u2);
	z1 = sqrt(-2.0 * log(u1)) * sin(tau * u2);
	return z0 * sigma + mu;
}



  //Hash info
  void createHash()
  {
    for(int i=0; i < N*N*N; i++)
      hash.push_back(HashData());
  }

  Vec3i coord(Vec3f a)
  {
    Vec3i tmp;
    for(int i=0;i<3;i++){
      tmp[i] = int((a[i]/2/BoxSize[i]+.5)*N);
      if(tmp[i]==N) tmp[i]=N-1;
    }
    return tmp;
  }

  void updateHash()
  {
    for (std::list<HashData>::iterator it = hash.begin(); it != hash.end(); ++it)
    {
      (it->vlist).clear();
      (it->flist).clear();
    };
    forall(const vertex &v, S)
    {
      std::list<HashData>::iterator it = hash.begin() ;
      std::advance(it,coord(v->pos)[0]+N*(coord(v->pos)[1]+N*coord(v->pos)[2]) );
      it->vlist.push_back((vertex*)&v);
    };
    forall(const filamin &f, F)
    { 
      std::list<HashData>::iterator it = hash.begin();
      std::advance(it,coord(f->pos)[0]+N*(coord(f->pos)[1]+N*coord(f->pos)[2]) );
      it->flist.push_back((filamin*)&f);
    };
    if(showhashinfo){
    for (std::list<HashData>::iterator it = hash.begin(); it != hash.end(); ++it){
      cout << it->vlist.size() << " " << it->flist.size() << endl;
      }
    cout << "___________________" << endl;
    };
    
  }
  



  //Drawing
  void initDraw(Viewer* viewer)
  {
    glPushMatrix();
    glLoadIdentity();

    glClearColor(0.0, 0.0, 0.0, 1.0);

    // Set lighting
    GLfloat lightc[] = {0.3f, 0.3f, 0.3f, 1.0f};

    glLightfv(GL_LIGHT0, GL_AMBIENT, lightc);
    glLightfv(GL_LIGHT1, GL_AMBIENT, lightc);
    glLightfv(GL_LIGHT2, GL_AMBIENT, lightc);
    glLightfv(GL_LIGHT3, GL_AMBIENT, lightc);

    glLightfv(GL_LIGHT0, GL_DIFFUSE, lightc);
    glLightfv(GL_LIGHT1, GL_DIFFUSE, lightc);
    glLightfv(GL_LIGHT2, GL_DIFFUSE, lightc);
    glLightfv(GL_LIGHT3, GL_DIFFUSE, lightc);

    glLightfv(GL_LIGHT0, GL_SPECULAR, lightc);
    glLightfv(GL_LIGHT1, GL_SPECULAR, lightc);
    glLightfv(GL_LIGHT2, GL_SPECULAR, lightc);
    glLightfv(GL_LIGHT3, GL_SPECULAR, lightc);

    GLfloat lpos0[] = {-1.0f, 1.0f, 1.0f, 0.0f};
    glLightfv(GL_LIGHT0, GL_POSITION, lpos0);

    GLfloat lpos1[] = {-1.0f, 1.0f, -1.0f, 0.0f};
    glLightfv(GL_LIGHT1, GL_POSITION, lpos1);

    GLfloat lpos2[] = {1.0f, 1.0f, 1.01f, 0.0f};
    glLightfv(GL_LIGHT2, GL_POSITION, lpos2);

    GLfloat lpos3[] = {1.0f, 1.0f, -1.0f, 0.0f};
    glLightfv(GL_LIGHT3, GL_POSITION, lpos3);

    glEnable(GL_LIGHT0);
    glEnable(GL_LIGHT1);
    glEnable(GL_LIGHT2);
    glEnable(GL_LIGHT3);
    glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, TRUE);

    glPopMatrix();

    float r = BoxSize.x();
    if(r < BoxSize.y())
      r = BoxSize.y();
    if(r < BoxSize.z())
      r = BoxSize.z();
    viewer->setSceneRadius(r * 10);
  }

  void draw()
  {
    if(WireFrame) {
      glDisable(GL_BLEND);
      glDisable(GL_LIGHTING);
      glLineWidth(1.0);

      drawForces();
      drawActin();
      drawFilamin();
    } else {
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
      glEnable(GL_LIGHTING);
      glDisable(GL_DEPTH_TEST);
      glBlendFunc(GL_SRC_ALPHA, GL_ONE);
      glEnable(GL_BLEND);
      glPolygonMode(GL_FRONT, GL_FILL);
      materials.useMaterial(0);
      glBegin(GL_TRIANGLES);
      forall(const vertex &v, S)
        if(v->draw)
          forall(const vertex &n, S.neighbors(v))
            if(S.edge(v, n)->draw) {
              vertex m = S.nextTo(v, n);
              while(m != n && !S.edge(v, m)->draw)
                m = S.nextTo(v, m);
              glVertex3fv(v->pos.c_data());
              glVertex3fv(n->pos.c_data());
              glVertex3fv(m->pos.c_data());
            }
      glEnd();

    }
    if(DrawBox) {
      glDisable(GL_BLEND);
      glDisable(GL_LIGHTING);
      glLineWidth(1.0);
      glColor3d(.4, .4, .4);
      glBegin(GL_LINES);      
      for(int i = -1; i <= 1; i+=2)
        for(int j = -1; j <= 1; j+=2)
          for(int k = -1; k <= 1; k+=2)
            glVertex3f(BoxSize.x() * i, BoxSize.y() * j, BoxSize.z() * k);
      for(int i = -1; i <= 1; i+=2)
        for(int k = -1; k <= 1; k+=2)
          for(int j = -1; j <= 1; j+=2)
            glVertex3f(BoxSize.x() * i, BoxSize.y() * j, BoxSize.z() * k);
      for(int k = -1; k <= 1; k+=2)
        for(int j = -1; j <= 1; j+=2)
          for(int i = -1; i <= 1; i+=2)
            glVertex3f(BoxSize.x() * i, BoxSize.y() * j, BoxSize.z() * k);
      glEnd();/*
      for(int n =0; n < N*N*N; n++)
	{
	  glBegin(GL_LINES);      
	  for(int i = -1; i <= 1; i+=2)
	  for(int j = -1; j <= 1; j+=2)
          for(int k = -1; k <= 1; k+=2)
	  glVertex3f(BoxSize.x() * (i+2*(n%N))/N - (N-1)/N*BoxSize.x(), BoxSize.y() * (j+2*(((n-n%N)/N)%N))/N - (N-1)/N*BoxSize.y(), BoxSize.z() * (k+2*((n-n%(N*N))/(N*N)%N))/N - (N-1)/N*BoxSize.z());
	  for(int i = -1; i <= 1; i+=2)
	  for(int k = -1; k <= 1; k+=2)
          for(int j = -1; j <= 1; j+=2)
	  glVertex3f(BoxSize.x() * (i+2*(n%N))/N - (N-1)/N*BoxSize.x(), BoxSize.y() * (j+2*(((n-n%N)/N)%N))/N - (N-1)/N*BoxSize.y(), BoxSize.z() * (k+2*((n-n%(N*N))/(N*N)%N))/N - (N-1)/N*BoxSize.z());
	  for(int k = -1; k <= 1; k+=2)
	  for(int j = -1; j <= 1; j+=2)
          for(int i = -1; i <= 1; i+=2)
	  glVertex3f(BoxSize.x() * (i+2*(n%N))/N - (N-1)/N*BoxSize.x(), BoxSize.y() * (j+2*(((n-n%N)/N)%N))/N - (N-1)/N*BoxSize.y(), BoxSize.z() * (k+2*((n-n%(N*N))/(N*N)%N))/N - (N-1)/N*BoxSize.z());
	  glEnd();
	}*/
    }
  }

  void drawForces()
  {
    if(DrawForce > 0) {
      glColor3f(0.5, 0.5, .5);
      glBegin(GL_LINES);
      forall(const vertex &v, S) {
	glVertex3fv(v->pos.c_data());
	glVertex3fv((v->pos + v->force * DrawForce).c_data());
      }
      glEnd();
    }
  }

  void drawActin()
  {
    glColor3d(0.0, 1.0, 0.0);
    glBegin(GL_LINES);
    forall(const vertex &v, S)
      if(v->draw)
	forall(const vertex &n, S.neighbors(v))
	  if(S.edge(v, n)->draw) {
	    glVertex3fv(v->pos.c_data());
	    glVertex3fv(n->pos.c_data());
	  }
    glEnd();
  }

  void drawFilamin()
  {
    glPointSize(3.0f);
    glEnable( GL_POINT_SMOOTH );
    glBegin( GL_POINTS );
    glColor3f( 0.95f, 0.207, 0.031f );    
    forall(const filamin &f, F)
      {
        if(f->draw) glVertex3f( f->pos.x(), f->pos.y(),f->pos.z() );
	
      }
    glEnd();
    
    glLineWidth(5.0);
    forall(const filamin &f, F)
      if(f->draw)
      {
	  glBegin(GL_LINES);	    
	  glVertex3fv((*(f->v[0]))->pos.c_data());
	  glVertex3fv(f->pos.c_data());
	  glVertex3fv((*(f->v[0]))->pos.c_data());
	  glVertex3fv((*(f->v[2]))->pos.c_data());
	  glVertex3fv((*(f->v[1]))->pos.c_data());
	  glVertex3fv(f->pos.c_data());
	  glVertex3fv((*(f->v[1]))->pos.c_data());
	  glVertex3fv((*(f->v[3]))->pos.c_data());
	  glEnd();
      }   
  }
  

  //old
  void findForces()
  {
    forall(const vertex &v, S) v->force=Vec3f(0,0,0);
    vertex n1;
    Vec3f a,b,c,d;
    float phi;
    forall(const vertex &v, S)
    {
      n1 = S.anyIn(v);
      forall(const vertex &n2, S.neighbors(v)){
              if(n2!=n1){
                c = S.target(S.edge(v,n1))->pos;
                d = S.source(S.edge(v,n1))->pos;
                a = normalizedSafe(c-d);
                c = S.target(S.edge(v,n2))->pos;
                d = S.source(S.edge(v,n2))->pos;
                b = normalizedSafe(c-d);
                phi = angleSafe(a, b);
                if(isnan(phi)) cout << "found NaN: findForces()" << endl;
                if(Debug)
                  cout << "abcd: " << a << " " << b << " " << c << " " << d << ". angle: " << phi << endl;
                // here ^ is cross product
                n1->force += normalizedSafe(a ^ (a ^ b)) * kBendActin * phi;
                n2->force += normalizedSafe((a ^ b) ^ b) * kBendActin * phi;
              }
              a = n2->pos;
              b = v->pos;
              v->force += normalizedSafe(a - b) * (norm(a - b) - S.edge(v, n2)->restlen) * kStretchActin;
      }
    }
    forall(const vertex &v, S)
      if(Debug)
        cout << "findForces(): " << v->pos << " " << v->vel << " " << v->force << endl;
  }
  
/*  
  void growFilamentRandom(float restlen)
  {
    int growthrate = 0;
    vertex n = S[rand() % S.size()];
    vertex tmp = n;
    forall(const vertex &w, S.neighbors(n))
      if(tmp != w && S.valence(w) == 1 && growthrate < 1) {
        vertex v;
        S.insert(v);
        v->pos.x() = unifRand(w->pos.x(),restlen);
        v->pos.y() = unifRand(w->pos.y(),restlen);
        v->pos.z() = unifRand(w->pos.z(),restlen);
        v->draw = true;
        insertSpring(v,w,restlen);
        growthrate++;
      } else if(growthrate < 1) {
        if(tmp != w) tmp=n;
         n=w;
      };
  }

  
  void growFilament(Vec3f r, float restlen)
  {
    int growthrate = 0;
    vertex n = S.any();
    forall(const vertex &w, S.neighbors(n))
      if(S.valence(w) == 1 && growthrate < 1) {
        vertex v;
        S.insert(v);
        v->pos.x() = r.x();
        v->pos.y() = r.y();
        v->pos.z() = r.z();
        v->draw = true;
        insertSpring(v,w,restlen);
        growthrate++;
      }
  }
  */
  
  void RungeKutta4()
  {
    vertex n1;
    Vec3f a,b,c,d;
    float phi;

    forall(const vertex &v, S)
        for(int i=0;i<8;i++) v->RK4[i]=Vec3f(0,0,0);

    findForces();
    //k0
    forall(const vertex &v, S){
      v->RK4[0] = Dt * (v->force - (DampA * v->vel));
      v->RK4[1] = Dt * v->vel;
    }

    //k1
    forall(const vertex &v, S){
      n1 = S.anyIn(v);
      forall(const vertex &n2, S.neighbors(v)){
              if(n2!=n1){
                c = S.target(S.edge(v,n1))->pos + .5*S.target(S.edge(v,n1))->RK4[1];
                d = S.source(S.edge(v,n1))->pos + .5*S.source(S.edge(v,n1))->RK4[1];
                a = normalizedSafe(c-d);
                c = S.target(S.edge(v,n2))->pos + .5*S.target(S.edge(v,n2))->RK4[1];
                d = S.source(S.edge(v,n2))->pos + .5*S.source(S.edge(v,n2))->RK4[1];
                b = normalizedSafe(c-d);
                phi = angleSafe(a, b);
                //cout << phi << endl;
                if(isnan(phi)) cout << "found NaN: RungeKutta4() k1" << endl;
                // here ^ is cross product
                n1->RK4[2] += normalizedSafe(a ^ (a ^ b)) * kBendActin * phi * Dt;
                n2->RK4[2] += normalizedSafe((a ^ b) ^ b) * kBendActin * phi * Dt;
              }
              a = n2->pos + .5*n2->RK4[1];
              b = v->pos + .5*v->RK4[1];
              v->RK4[2] += normalizedSafe(a - b) * (norm(a - b) - S.edge(v, n2)->restlen) * kStretchActin * Dt;
              v->RK4[2] -= DampA * (v->vel + .5*v->RK4[0])*Dt;
      }
      if(Debug)
        cout << "rk4[2]: " << v->pos << " " << v->RK4[2] << endl;
    }
    forall(const vertex &v, S){
      v->RK4[3] = Dt * (v->vel + .5*v->RK4[0]);
      if(Debug)
        cout << "rk4[3]: " << v->pos << " " << v->RK4[3] << endl;
    }
    //k2
    forall(const vertex &v, S){
      n1 = S.anyIn(v);
      forall(const vertex &n2, S.neighbors(v)){
        if(n2!=n1){
            c = S.target(S.edge(v,n1))->pos + .5*S.target(S.edge(v,n1))->RK4[3];
            d = S.source(S.edge(v,n1))->pos + .5*S.source(S.edge(v,n1))->RK4[3];
            a = normalizedSafe(c-d);
            c = S.target(S.edge(v,n2))->pos + .5*S.target(S.edge(v,n2))->RK4[3];
            d = S.source(S.edge(v,n2))->pos + .5*S.source(S.edge(v,n2))->RK4[3];
            b = normalizedSafe(c-d);
            phi = angleSafe(a, b);
            if(isnan(phi)) cout << "found NaN: RungeKutta4() k2" << endl;
            // here ^ is cross product
            n1->RK4[4] += normalizedSafe(a ^ (a ^ b)) * kBendActin * phi * Dt;
            n2->RK4[4] += normalizedSafe((a ^ b) ^ b) * kBendActin * phi * Dt;
          }
        a = n2->pos + .5*n2->RK4[3];
        b = v->pos + .5*v->RK4[3];
        v->RK4[4] += normalizedSafe(a - b) * (norm(a - b) - S.edge(v, n2)->restlen) * kStretchActin * Dt;
        v->RK4[4] -= DampA * (v->vel + .5*v->RK4[2])*Dt;
      }
      if(Debug)
        cout << "rk4[4]: " << v->pos << " " << v->RK4[4] << endl;
    }
    forall(const vertex &v, S){
      v->RK4[5] = Dt * (v->vel + .5*v->RK4[2]);
      if(Debug)
        cout << "rk4[5]: " << v->pos << " " << v->RK4[5] << endl;
    }
   //k3
    forall(const vertex &v, S){
      if(Debug)
        cout << "rk4[4]: " << v->pos << " " << v->RK4[4] << endl;
      n1 = S.anyIn(v);
      forall(const vertex &n2, S.neighbors(v)){
            if(Debug)
        cout << "rk4[4]: " << v->pos << " " << v->RK4[4] << endl;
            if(n2!=n1){
            c = S.target(S.edge(v,n1))->pos + S.target(S.edge(v,n1))->RK4[3];
            d = S.source(S.edge(v,n1))->pos + S.source(S.edge(v,n1))->RK4[3];
            a = normalizedSafe(c-d);
            c = S.target(S.edge(v,n2))->pos + S.target(S.edge(v,n2))->RK4[3];
            d = S.source(S.edge(v,n2))->pos + S.source(S.edge(v,n2))->RK4[3];
            b = normalizedSafe(c-d);
	    phi = angleSafe(a, b);
            if(isnan(phi)) cout << "found NaN: RungeKutta4() k3" << endl;
            if(Debug)
              cout << "rk6,abcd: " << a << " " << b << " " << c << " " << d << ". angle: " << phi << endl;
            // here ^ is cross product
            n1->RK4[6] += normalizedSafe(a ^ (a ^ b)) * kBendActin * phi * Dt;
            n2->RK4[6] += normalizedSafe((a ^ b) ^ b) * kBendActin * phi * Dt;
            if(Debug)
              cout << "rk6,angularF: " << normalizedSafe(a ^ (a ^ b)) * kBendActin * phi * Dt << " " << normalizedSafe((a ^ b) ^ b) * kBendActin * phi * Dt << endl;
          }
        a = n2->pos + n2->RK4[5];
        b = v->pos + v->RK4[5];
        v->RK4[6] += normalizedSafe(a - b) * (norm(a - b) - S.edge(v, n2)->restlen) * kStretchActin * Dt;
        v->RK4[6] -= DampA * (v->vel + v->RK4[4])*Dt;
              if(Debug){
                cout << "rk6,step: " << normalizedSafe(a - b) * (norm(a - b) - S.edge(v, n2)->restlen) * kStretchActin * Dt << " " << DampA * (v->vel + v->RK4[4])*Dt << endl;
                cout << "why?!: " << DampA << " " << v->vel << " " << v->RK4[4] << " " << Dt << endl;
              }

      }
      if(Debug)
        cout << "rk4[6]: " << v->pos << " " << v->RK4[6] << endl;
    }

    forall(const vertex &v, S)
      v->RK4[7] = Dt * (v->vel + v->RK4[4]);

  //final step of Runge-Kutta
    forall(const vertex &v, S){
      v->vel += (v->RK4[0]+2*v->RK4[2]+2*v->RK4[4]+v->RK4[6])/6;
      v->pos += (v->RK4[1]+2*v->RK4[3]+2*v->RK4[5]+v->RK4[7])/6;
      if(Debug)
        cout << "rk8(): " << v->pos << " " << v->vel << " " << v->force << endl;
      //constraints(v);
    }
      steps++;
      timing += Dt;
  }

};

// Define the model to be used
DEFINE_MODEL(myModel);
