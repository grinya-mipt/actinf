[Main]
Debug: false
Dt: .002                                 // Timestep
BoxSize: 100 100 100			// Size of box
CellSize: 2				// Cell size
WireFrame: true				// Draw wireframe or shaded
DrawNormals: 0				// Size to draw normals
DrawForce: 0				// Draw force vector
DrawBox: true				// Draw the box
FrameSteps: 10				// Steps per frame
PrintStats: false			// Print out stats
Parallel: false
SeedRandom: 0
TimeSolver: 0    			// 0 for forward euler, 1 for runge kutta
HashSize: 10
ShowHashInfo: false
Constraints: true

[Actin]
kStretchActin: 1000			// Regular stretch constant
kBendActin: 500				// Regular bend constant
Damp: 10			// Spring dampening
RestLength: 1				// Spring rest length
PrintForce: false			// Print out force vectors
Gravity: 0				// Gravity strength
Friction: 50				// Friction when it hits the walls
Thresold: .004				// Threshold velocity for surface tension
RestLenght: 10
Variance: 0				//Stochastic force variance
Number of filaments: 1000
Number for growth: 100000

[Filamin]
kStretchFilamin: 500				
kBendFilamin: 1000	
RestLenghtFilamin: 5
Damp: 10
Variance: 0

[Output]
OutputNumericsFileName: out_numerics.txt
