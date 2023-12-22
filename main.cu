#include "nn_utils/shape.hh"
#include <iostream>

int main(int argc, char const *argv[])
{
    size_t a=30, b=30;
    Shape s = Shape(a, b);

    printf("%zu, %zu \n", s.x, s.y);

    return 0;
}
