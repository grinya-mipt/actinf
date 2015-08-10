[Main]
Debug: false
Dt: .05                                 // Timestep
BoxSize: 500 500 500			// Size of box
CellSize: 1				// Cell size
WireFrame: true				// Draw wireframe or shaded
DrawNormals: 0				// Size to draw normals
DrawForce: 0				// Draw force vector
DrawBox: true				// Draw the box
FrameSteps: 10				// Steps per frame
PrintStats: false			// Print out stats
Parallel: false
SeedRandom: 0
TimeSolver: 1    			// 0 for forward euler, 1 for runge kutta

[Spring]
KR: 800					// Regular stretch constant
KB: 400					// Regular bend constant
Damp: 100			// Spring dampening
RestLength: 1				// Spring rest length
PrintForce: false			// Print out force vectors
Gravity: 0				// Gravity strength
Friction: 50				// Friction when it hits the walls
Thresold: .004				// Threshold velocity for surface tension
RestLenght: 10
Variance: 100
Number of filaments: 100
Number for growth: 1000000
