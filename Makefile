OBJECTS=model.o cuda.o

default: model.vve

standalone: model

include $(shell vveinterpreter --makefile)
# Uncomment the next line to compile with debug options
#CXXFLAGS=$(CXXFLAGS_DEBUG)
# Add extra compilation options here
CXXFLAGS+=
# Add extra includes here
INCLUDES+=-isystem /usr/local/cuda/include -I.
# Add extra libraries here
LIBS+=-lcudart -lcuda -lgomp
# Add extra linking options here
#    for compiling the model as a library (.vve)
LD_SO_FLAGS+=-L/usr/local/cuda/lib64 -L/usr/local/lib -L/usr/lib
#    for compiling the model as a stand-alone program
LD_EXE_FLAGS+=

model.o: model.moc

model.vve: $(OBJECTS)

model${EXESUFFIX}: $(OBJECTS)

#LIBS = -L/usr/local/lib -L/usr/lib -L/usr/local/cuda/lib64

%.o: %.cu $(wildcard %.h)
	nvcc -O3 $(INCLUDES) -I /usr/local/cuda/include --ptxas-options=-v --compiler-options=-fPIC -c $< 

model.so: $(OBJ)
	$(CXX) $(OBJ) -o $@ -shared -Wl,-soname=$@ $(LIBS) -lvvelib -lGLU -lQtCore -lQtGui -lQtOpenGL -lQtXml -lQtUiTools -lcudart -lcuda -lgomp

run:
	vveinterpreter model.vve

clean:
	$(RM) *.o model model.vve
