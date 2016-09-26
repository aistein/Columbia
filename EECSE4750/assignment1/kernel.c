#include <stdio.h>
#include <string.h>
int main()
{
       /* Our first simple C basic program */
       printf("Hello World! \n");

       char string[32] = "Hello World!\n";
       char *ptr = string;
       char x = ptr[0];
       printf("char is %c\n", x);
       printf("Length of string: %zu\n", strlen(string));

       return 0;
}
