#include<stdio.h>

void bar(int x, int y);
void foo(int a, int b, int c){
    int d, e, f;
    bar(7,12);
}

int main(){
    foo(1,2,3);
    return 0;
}

void bar(int x, int y){
    printf("nums: %d, %d \n", x,y);
}
