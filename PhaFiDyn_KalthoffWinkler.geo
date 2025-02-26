E = 190e9;
nu = 0.3;
Gc = 2.213e4;
sigc = (27/256*E*Gc/1.95e-4)^0.5; // 1.07e9;
lc = 27/256*E*Gc/sigc/sigc;
Printf("PF >> sigc = %f", sigc);
Printf("PF >> lc = %f", lc);
// lc = 1.95e-4;

L = 0.10;
B = 0.10;

Ratio = 5e-2;
hc = lc/Ratio;
hc = 0.0035;

Ratio = 2;
hf = lc/Ratio;

Printf("PF >> hc = %f", hc);
Printf("PF >> hf = %f", hf);

// generation du rectangle

Point(1)={0, 0, 0, hc};

Point(2)={L,0.,0., hc};

Point(3)={L, B, 0., hc};

Point(4)={0, B, 0., hc};

k = lc*2;
Printf("PF >> Notch = %f", k);

Point(5)={L/2, 0.025+k, 0., hf};
Point(6)={L/2, 0.025, 0., hf};
Point(15)={L/2-0.005, 0.025+k, 0., hf};

Point(16)={L/2+0.005, 0.025, 0., hf};

Point(7)={0, 0.025+k, 0., hc};
Point(8)={0, 0.025, 0., hc};


Point(17)={L-0.01, B, 0., hf};

Point(18)={L-(0.03-0.01), B, 0., hf};


//+
Line(1) = {1, 2};
//+
Line(2) = {2, 3};
//+
Line(3) = {3, 17};
//+
Line(4) = {17, 18};
//+
Line(5) = {18, 4};
//+
Line(6) = {4, 7};
//+
Line(7) = {8, 1};
//+
Line(8) = {8, 6};
//+
Line(9) = {6, 5};
//+
Line(10) = {5, 15};
//+
Line(11) = {15, 7};
//+
Line(12) = {6, 16};
//+
Line(13) = {16, 17};
//+
Line(14) = {18, 15};

//+
Curve Loop(1) = {5, 6, -11, -14};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {14, -10, -9, 12, 13, 4};
//+
Plane Surface(2) = {2};
//+
Curve Loop(3) = {2, 3, -13, -12, -8, 7, 1};
//+
Plane Surface(3) = {3};
//+


Physical Surface(1000) = {1, 3};

Physical Surface(2000) = {2};
//+
Physical Curve(100) = {1}; //Sym y
//+
Physical Curve(200) = {7}; // load

Physical Curve(300) = {11, 8, 9, 10}; // Crack