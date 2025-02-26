lc = .25e-3;
E = 32e9;
Gc = 3;
sigc = (27/256*E*Gc/lc)^0.5;
Printf("PF-CZM >> sigc [Pa] = %f", sigc);
Printf("PF-CZM >> sigc [Pa] = %f", 6.36e7);

lc = 5e-4;
Printf("PF-CZM >> lc [mm] = %f", lc);
Printf("PF-CZM >> 2D [mm] = %f", 3.14*lc);

//Maillage homogène Ratio={2, 4, 5, 6 , 7, 10, 15, 20} 

Ratio = 4.;     
h = lc/Ratio;
Ratio = 4;     
hf = lc/Ratio;

Printf("PF >> Ratio = %f", Ratio);
Printf("PF>> hc [mm] = %f", h);
Printf("PF >> hf [mm] = %f", hf);

L = 0.100;      // Longueur
W = 0.04;    // largeur

load = lc;  //loading Zone


Point(01) = {    -L/2,     -W/2,     0,      h   };
Point(02) = {     L/2,     -W/2,     0,      hf   };
Point(03) = {     L/2,      W/2,     0,      hf   };
Point(04) = {    -L/2,      W/2,     0,      h   };

//Point(05) = {     L/4,      W/2,     0,      hf   };
//Point(06) = {    L/4,      -W/2,     0,      hf   };


//Notch

a     = L/2;    b = 5e-4;

Point(10) = {   -(L/2)  ,       -b/2    ,        0,       hf      };
Point(11) = {   0 ,    b/2 ,        0,       lc/2      };

Point(13) = {    -(L/2)     ,     b/2   ,        0,       hf      };


Point(14) = {   0,       -b/2    ,        0,       lc/2      };

//+
Line(1) = {1, 2};
//+
Line(2) = {2, 3};
//+
Line(3) = {3, 4};
//+
Line(6) = {14, 11};
Line(7) = {14, 10};

//+
Line(8) = {11, 13};
//+
Line(9) = {13, 4};
//+
//+
Line(11) = {1, 10};

//+

Curve Loop(1) = {3, -9, -8, -6, 7, -11, 1, 2};
//+
Plane Surface(1) = {1};

Physical Surface(1000) = {1};//+


Physical Curve(100) = {3}; // Top
Physical Curve(200) = {1}; // Bot
Physical Curve(300) = {8, 6, 7}; // Fissure
Physical Curve(400) = {11, 9, 2}; // other edge
//+

