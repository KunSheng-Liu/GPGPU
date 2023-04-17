#define min2(a,b) (((a)<(b))?(a):(b))
#define min3(x,y,z) (((x)<(y) && (x)<(z))?(x):(min2((y),(z))))
#define min4(w,x,y,z) ((min2(w,x) < min2(y,z)) ? min2(w,x) : min2(y,z))