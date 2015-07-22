[Main]
Dt: .002				// Timestep
CubeSize: 4 4 8 			// Size of cube *2+1
BoxSize: 2 2 2				// Size of box
CellSize: .1				// Cell size
WireFrame: false			// Draw wireframe or shaded
DrawNormals: 0				// Size to draw normals
DrawForce: 0				// Draw force vector
DrawBox: true				// Draw the box
FrameSteps: 100				// Steps per frame
PrintStats: false			// Print out stats
Parallel: true

[Spring]
KR: 200					// Regular spring constant
Damp: .3				// Spring dampening
RestLength: 1				// Spring rest length
PrintForce: false			// Print out force vectors
Gravity: .15				// Gravity strength
Friction: 50				// Friction when it hits the walls
Thresold: .004				// Threshold velocity for surface tension
