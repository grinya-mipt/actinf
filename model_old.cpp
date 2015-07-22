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
  Vec3f pos;					// position
  Vec3f vel;					// velocity
  Vec3f force;					// Uniform force on points

  bool border1;
  bool border2;
  bool draw;					// Draw vertex

  // Constructor, set initial values
  VertexData() : pos(0,0,0), vel(0,0,0), force(0,0,0),border1(false),border2(false), draw(false) {}
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
  float Dt;					// Timestep
  Vec3f BoxSize;				// Size of box
  float CellSize;				// Size of cells
  bool WireFrame;				// Draw wireframe or shaded
  float DrawNormals;				// Size to draw normals
  float DrawForce;				// Size to force vectors
  bool DrawBox;					// Draw the box
  float FrameSteps;				// Steps per frame
  bool PrintStats;				// Print out stats
  bool Parallel;				// Use Cuda

  float KR;					// Regular spring constant
  float Damp;					// Spring dampening
  bool PrintForce;				// Print out force vectors
  float Gravity;				// Gravity strength
  float Friction;				// Friction when cube hits wall
  float Threshold;				// Threshold velocity under which vertices on surface don't move

private:
  // Model variables
  vvgraph S;					// Graph vertices
  std::vector<vertex> E;			// Exterior verticesce

  util::Palette palette;			// Colors
  util::Materials materials;			// Materials

  int steps;					// Step count
  double time;					// Time counter
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

    parms("Spring", "KR", KR);
    parms("Spring", "Damp", Damp);
    parms("Spring", "PrintForce", PrintForce);
    parms("Spring", "Gravity", Gravity);
    parms("Spring", "Friction", Friction);
    parms("Spring", "Thresold", Threshold);
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

    // Registering the configuration files
    registerFile("pal.map");
    registerFile("view.v");
        
	vertex v;
    S.insert(v);
    v->pos.x() = 0;
    v->pos.y() = BoxSize[1]*2*(double(rand())/double(RAND_MAX)-0.5);
    v->pos.z() = -BoxSize[2]*CellSize;
    v->draw = true;
    v->border2 = true;
    
    vertex w;
    S.insert(w);
	w->pos.x() = 0;
    w->pos.y() = BoxSize[1]*2*(double(rand())/double(RAND_MAX)-0.5);
    w->pos.z() = BoxSize[2]*CellSize;            
    w->draw = true;
    w->border1 = true;
    
    S.insertEdge(v, w);
    S.edge(v, w)->draw = true;
    S.edge(v, w)->restlen = norm(v->pos - w->pos);
    
    time = 0;
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
	// Make filaments
	        
	vertex v;
    S.insert(v);
    v->pos.x() = 0;
    v->pos.y() = BoxSize[1]*2*(double(rand())/double(RAND_MAX)-0.5);
    v->pos.z() = -BoxSize[2]*CellSize;
    v->draw = true;
    v->border2 = true;
    
    vertex w;
    S.insert(w);
	w->pos.x() = 0;
    w->pos.y() = BoxSize[1]*2*(double(rand())/double(RAND_MAX)-0.5);
    w->pos.z() = BoxSize[2]*CellSize;            
    w->draw = true;
    w->border1 = true;
    
    S.insertEdge(v, w);
    S.edge(v, w)->draw = true;
    S.edge(v, w)->restlen = 1;//norm(v->pos - w->pos);
	
	S.insertEdge(w, v);
	S.edge(w, v)->draw = true;
    S.edge(w, v)->restlen = 1;//norm(v->pos - w->pos);


    if(Parallel) {
      // Call cuda to do the work
      massSpring(nhbd.idx, pos.idx, vel.idx, force.idx, restlen.idx, BoxSize.c_data(), 0/*gravity.c_data()*/, KR, Damp, Friction, Threshold, Dt, FrameSteps);
      // Get data back from cuda
      pos.copyToHost();

      // Update time
      steps += FrameSteps;
      time += Dt * FrameSteps;
    } else {
      for(int i = 0; i < FrameSteps; i++) {
        // Find force on vertivces
        forall(const vertex &v, S)
		  v->force=Vec3f(0,0,0); 
        forall(const vertex &v, S) {
          // Start with pulling force
          //cout << time << endl;
          //if(v->border1 and time <100.0)
            //v->force = Vec3f(0,0,1);

           // Force in walls
          forall(const vertex &n, S.neighbors(v)) {
            // Spring forces
            Vec3f dir = n->pos - v->pos;
            float len = norm(dir);
            dir /= len;
            v->force += dir * (len - S.edge(v, n)->restlen) * KR;
            cout << dir * (len - S.edge(v, n)->restlen) * KR << endl;
          }
        }
        forall(const vertex &v, S) {  
          // Move points
          if(!v->border2){
            v->vel += Dt * (v->force - (Damp * v->vel));
            v->pos += Dt * v->vel;
		  }
		  /*
          // Hit the box
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
          }
		  */
          steps++;
          time += Dt;
        }
      }
    }

    if(PrintStats && !(steps % 100))
      cout << "Explicit step. Steps: " << steps << ". Time: " << time << endl;
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
      forall(const vertex &v, S){
        if(v->draw)
          forall(const vertex &n, S.neighbors(v))
            if(S.edge(v, n)->draw) {
              glVertex3fv(v->pos.c_data());
              glVertex3fv(n->pos.c_data());
      
            }
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
      forall(const vertex &v, E)
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
