/* ************************************************************************************************
 * Macro
 * ************************************************************************************************
 */
#define str(x) #x

#define GET_FUNCTION( _1, _2, function, ... ) function 
/* Exit when condition == false */
#define ASSERT( ... )   GET_FUNCTION( __VA_ARGS__, ASSERT_2, ASSERT_1 ) (__VA_ARGS__)
// #define DEBUG( ... )    GET_FUNCTION( __VA_ARGS__, DEBUG_2, DEBUG_1 )   (__VA_ARGS__)

/* Assert function with print message */
#define ASSERT_1( condition ) { if (!(condition)) exit (1); }
#define ASSERT_2( condition, ... ) {                                           \
    if (!(condition)) {                                                        \
        std::cout << __FILE__  << ": " << __LINE__ << ": " <<  __func__        \
                  << ": " << __VA_ARGS__ << std::endl;                         \
    exit (1);} }

/* ************************************************************************************************
 * Math
 * ************************************************************************************************
 */
#define min2(a,b) (((a)<(b))?(a):(b))
#define min3(x,y,z) (((x)<(y) && (x)<(z))?(x):(min2((y),(z))))
#define min4(w,x,y,z) ((min2(w,x) < min2(y,z)) ? min2(w,x) : min2(y,z))