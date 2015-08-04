[Main]
Debug: false
Dt: .00002	//0.0002			// Timestep
BoxSize: 2 2 2				// Size of box
CellSize: 1				// Cell size
WireFrame: true			// Draw wireframe or shaded
DrawNormals: 0				// Size to draw normals
DrawForce: 1				// Draw force vector
DrawBox: true				// Draw the box
FrameSteps: 10				// Steps per frame
PrintStats: false			// Print out stats
Parallel: false
SeedRandom: 0
TimeSolver: 1    // 0 for forward euler, 1 for runge kutta (can be changed between time steps)

[Spring]
KR: 0					// Regular spring constant
KB: 20
Damp: 0					// Spring dampening
RestLength: 1				// Spring rest length
PrintForce: false			// Print out force vectors
Gravity: 0				// Gravity strength
Friction: 50				// Friction when it hits the walls
Thresold: .004				// Threshold velocity for surface tension
RestLenght: 1


