#include <vve.h>

#include <string>
#include <math.h>

#include <geometry/geometry.h>
#include <algorithms/insert.h>
#include <util/parms.h>
#include <util/palette.h>
#include <util/materials.h>

#include "dlobject.h"

#include <QFile>

using std::cout;
using std::endl;
using std::string;

typedef util::Vector<3, int> Vec3i;
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

// Type of the VV graph
typedef VVGraph<VertexData, EdgeData> vvgraph;

// Type of a vertex
typedef vvgraph::vertex_t vertex;

// Type of an edge
typedef vvgraph::edge_t edge;

// Type of distributed object neighborhood object
typedef DNObj<vvgraph> DNObjT;

// Class definition of model
class myModel : public Model
{
public:
  // Parameters
  float Dt;					// timingstep
  //Vec3i CubeSize;				// Size of cube
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

  float KR;					// Regular spring constant
  float KB;
  float Damp;					// Spring dampening
  bool PrintForce;				// Print out force vectors
  float Gravity;				// Gravity strength
  float Friction;				// Friction when cube hits wall
  float Threshold;				// Threshold velocity under which vertices on surface don't move
  float RestLenght;			

private:
  // Model variables
  vvgraph S;					// Graph vertices
  std::vector<vertex> E;			// Exterior vertices

  util::Palette palette;			// Colors
  util::Materials materials;			// Materials

  int steps;					// Step count
  float timing;					// timing counter
  Vec3f gravity;				// Gravity vector

  DNObjT nhbd;					// Distributed objects
  DVObj<vvgraph, DNObjT, Vec3f> pos;
  DVObj<vvgraph, DNObjT, Vec3f> vel;
  DVObj<vvgraph, DNObjT, Vec3f> force;
  DEObj<vvgraph, DNObjT, float> restlen;

public:
  void readParms()
  {
    util::Parms parms("view.v");

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

    parms("Spring", "KR", KR);
    parms("Spring", "KB", KB);
    parms("Spring", "Damp", Damp);
    parms("Spring", "PrintForce", PrintForce);
    parms("Spring", "Gravity", Gravity);
    parms("Spring", "Friction", Friction);
    parms("Spring", "Thresold", Threshold);
    parms("Spring", "RestLenght", RestLenght);
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

  // Constructor, build mesh
  myModel(QObject *parent): Model(parent), palette("pal.map"), materials("material.mat"), nhbd(&S), pos(&nhbd, offsetof(VertexData, pos)),
              vel(&nhbd, offsetof(VertexData, vel)), force(&nhbd, offsetof(VertexData, force)), restlen(&nhbd, offsetof(EdgeData, restlen))
  {
    // Read the parameters
    readParms();
    srand(SeedRandom==0?time(0):SeedRandom);

    
    Vec3f r,r1,r2;

    for(int i=0;i<1;i++) // SPECIFY NUMBER OF Filaments
      {
	//r1=Vec3f(unifRand(.0, 2.) * BoxSize[0],unifRand(.0, 2.) * BoxSize[1],unifRand(.0, 2.) * BoxSize[2]) * CellSize;
	//r2=Vec3f(unifRand(.0, 2.) * BoxSize[0],unifRand(.0, 2.) * BoxSize[1],unifRand(.0, 2.) * BoxSize[2]) * CellSize;
	r1 = Vec3f(0.,1.,0.);
	r2 = Vec3f(-1.,-1.,0.);
        RestLenght = norm(r1-r2);
	createFilament(r1,r2,RestLenght);
      }
    //r = Vec3f(unifRand(.0, 2.) * BoxSize[0],unifRand(.0, 2.) * BoxSize[1],unifRand(.0, 2.) * BoxSize[2]) * CellSize; 
    r = Vec3f(1.,-1.,0.);
    growFilament(r, RestLenght);
						
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
      massSpring(nhbd.idx, pos.idx, vel.idx, force.idx, restlen.idx, BoxSize.c_data(), gravity.c_data(), KR, Damp, Friction, Threshold, Dt, FrameSteps);
      // Get data back from cuda
      pos.copyToHost();

      // Update timing
      steps += FrameSteps;
      timing += Dt * FrameSteps;
    } else {
      for(int i = 0; i < FrameSteps; i++) {
        findForces();       
	RungeKhutta4();
      }
    }

    if(PrintStats && !(steps % 100))
      cout << "Explicit step. Steps: " << steps << ". timing: " << timing << endl;
  }
  
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
	    a = normalized(c-d);
	    c = S.target(S.edge(v,n2))->pos;
	    d = S.source(S.edge(v,n2))->pos;
	    b = normalized(c-d);
	    phi = angle(a, b);
	    // here ^ is cross product
	    n1->force += normalized(a ^ (a ^ b)) * KB * phi;
	    n2->force += normalized((a ^ b) ^ b) * KB * phi;
	  }
	a = n2->pos;
	b = v->pos;
	v->force += normalized(a - b) * (norm(a - b) - S.edge(v, n2)->restlen) * KR;
      }
    }
  }

  void findVel()
  {
    forall(const vertex &v, S)
      v->vel += Dt * (v->force - (Damp * v->vel));
  }
  
  void RungeKhutta4()
  {
    vertex n1;
    Vec3f a,b,c,d;
    float phi;

    findForces();
    //k0
    forall(const vertex &v, S){
      v->RK4[0] = Dt * (v->force - (Damp * v->vel));
      v->RK4[1] = Dt * v->vel;
    }
    //k1
    forall(const vertex &v, S){
      n1 = S.anyIn(v);
      forall(const vertex &n2, S.neighbors(v)){
	if(n2!=n1){
	    c = S.target(S.edge(v,n1))->pos + .5*S.target(S.edge(v,n1))->RK4[1];
	    d = S.source(S.edge(v,n1))->pos + .5*S.source(S.edge(v,n1))->RK4[1];
	    a = normalized(c-d);
	    c = S.target(S.edge(v,n2))->pos + .5*S.target(S.edge(v,n2))->RK4[1];
	    d = S.source(S.edge(v,n2))->pos + .5*S.source(S.edge(v,n2))->RK4[1];
	    b = normalized(c-d);
	    phi = angle(a, b);
	    // here ^ is cross product
	    n1->RK4[2] += normalized(a ^ (a ^ b)) * KB * phi * Dt;
	    n2->RK4[2] += normalized((a ^ b) ^ b) * KB * phi * Dt;
	  }
	a = n2->pos + .5*n2->RK4[1];
	b = v->pos + .5*v->RK4[1];
	v->RK4[2] += normalized(a - b) * (norm(a - b) - S.edge(v, n2)->restlen) * KR * Dt;
	v->RK4[2] -= Damp * (v->vel + .5*v->RK4[0])*Dt;
      }
    }
    forall(const vertex &v, S)
      v->RK4[3] = Dt * (v->vel + .5*v->RK4[0]);
    //k2
    forall(const vertex &v, S){
      n1 = S.anyIn(v);
      forall(const vertex &n2, S.neighbors(v)){
	if(n2!=n1){
	    c = S.target(S.edge(v,n1))->pos + .5*S.target(S.edge(v,n1))->RK4[3];
	    d = S.source(S.edge(v,n1))->pos + .5*S.source(S.edge(v,n1))->RK4[3];
	    a = normalized(c-d);
	    c = S.target(S.edge(v,n2))->pos + .5*S.target(S.edge(v,n2))->RK4[3];
	    d = S.source(S.edge(v,n2))->pos + .5*S.source(S.edge(v,n2))->RK4[3];
	    b = normalized(c-d);
	    phi = angle(a, b);
	    // here ^ is cross product
	    n1->RK4[4] += normalized(a ^ (a ^ b)) * KB * phi * Dt;
	    n2->RK4[4] += normalized((a ^ b) ^ b) * KB * phi * Dt;
	  }
	a = n2->pos + .5*n2->RK4[3];
	b = v->pos + .5*v->RK4[3];
	v->RK4[4] += normalized(a - b) * (norm(a - b) - S.edge(v, n2)->restlen) * KR * Dt;
	v->RK4[4] -= Damp * (v->vel + .5*v->RK4[2])*Dt;
      }
    }
    forall(const vertex &v, S)
      v->RK4[5] = Dt * (v->vel + .5*v->RK4[2]);
   //k3
    forall(const vertex &v, S){
      n1 = S.anyIn(v);
      forall(const vertex &n2, S.neighbors(v)){
	if(n2!=n1){
	    c = S.target(S.edge(v,n1))->pos + S.target(S.edge(v,n1))->RK4[3];
	    d = S.source(S.edge(v,n1))->pos + S.source(S.edge(v,n1))->RK4[3];
	    a = normalized(c-d);
	    c = S.target(S.edge(v,n2))->pos + S.target(S.edge(v,n2))->RK4[3];
	    d = S.source(S.edge(v,n2))->pos + S.source(S.edge(v,n2))->RK4[3];
	    b = normalized(c-d);
	    phi = angle(a, b);
	    // here ^ is cross product
	    n1->RK4[6] += normalized(a ^ (a ^ b)) * KB * phi * Dt;
	    n2->RK4[6] += normalized((a ^ b) ^ b) * KB * phi * Dt;
	  }
	a = n2->pos + n2->RK4[5];
	b = v->pos + v->RK4[5];
	v->RK4[6] += normalized(a - b) * (norm(a - b) - S.edge(v, n2)->restlen) * KR * Dt;
	v->RK4[6] -= Damp * (v->vel + v->RK4[4])*Dt;
      }
    }
    forall(const vertex &v, S)
      v->RK4[7] = Dt * (v->vel + v->RK4[4]); 
  //final step of Runge-Kutta
    forall(const vertex &v, S){
      v->vel += (v->RK4[0]+2*v->RK4[2]+2*v->RK4[4]+v->RK4[6])/6;
      v->pos += (v->RK4[1]+2*v->RK4[3]+2*v->RK4[5]+v->RK4[7])/6;
    }
      steps++;
      timing += Dt;
  }
  
  void movePoints()
  {
    forall(const vertex &v, S) 
    {
      v->vel += Dt * (v->force - (Damp * v->vel));
      v->pos += Dt * v->vel;       
          // Hit the box
	  /*
          if(v->pos.x() > BoxSize.x()) {
            v->pos.x() = BoxSize.x();
            v->vel -= Dt * v->vel * Friction * v->vel.x() * v->vel.x();
            v->vel.x() = 0;
          }
          if(v->pos.x() < -BoxSize.x()) {
            v->pos.x() = -BoxSize.x();
            v->vel -= Dt * v->vel * Friction * v->vel.x() * v->vel.x();
            v->vel.x() = 0;
          }
          if(v->pos.y() > BoxSize.y()) {
            v->pos.y() = BoxSize.y();
            v->vel -= Dt * v->vel * Friction * v->vel.y() * v->vel.y();
            v->vel.y() = 0;
          }
          if(v->pos.y() < -BoxSize.y()) {
            v->pos.y() = -BoxSize.y();
            v->vel -= Dt * v->vel * Friction * v->vel.y() * v->vel.y();
            v->vel.y() = 0;
          }
          if(v->pos.z() > BoxSize.z()) {
            v->pos.z() = BoxSize.z();
            v->vel -= Dt * v->vel * Friction * v->vel.z() * v->vel.z();
            v->vel.z() = 0;
          }
          if(v->pos.z() < -BoxSize.z()) {
            v->pos.z() = -BoxSize.z();
            v->vel -= Dt * v->vel * Friction * v->vel.z() * v->vel.z();
            v->vel.z() = 0;
          }*/

      steps++;
      timing += Dt;
    }	    
  }
  
  void createFilament(Vec3f r1, Vec3f r2, float restlen)
  {
    vertex v;
    S.insert(v);
    v->pos.x() = r1.x();
    v->pos.y() = r1.y();
    v->pos.z() = r1.z();
    v->draw = true;
    v->barb = true;
    vertex w;
    S.insert(w);
    w->pos.x() = r2.x();
    w->pos.y() = r2.y();
    w->pos.z() = r2.z();
    w->draw = true;      
    insertSpring(v,w,restlen);
  }
  
  void growFilament(Vec3f r,float restlen)
  {
    int growthrate = 0;
    vertex n= S.any();
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

  // Initialize drawing
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
    viewer->setSceneRadius(r * 2);
  }


  void draw(/*Viewer* viewer*/)
  {
    if(WireFrame) {
      glDisable(GL_BLEND);
      glDisable(GL_LIGHTING);
      glLineWidth(1.0);
      if(DrawForce > 0) {
        glColor3f(0.5, 0.5, .5);
        glBegin(GL_LINES);
        forall(const vertex &v, S) {
          glVertex3fv(v->pos.c_data());
          glVertex3fv((v->pos + v->force * DrawForce).c_data());
        }
        glEnd();
      }

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
      glEnd();
    }
  }
};

// Define the model to be used
DEFINE_MODEL(myModel);
