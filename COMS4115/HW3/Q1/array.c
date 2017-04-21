#include <stdio.h>
#include <stdlib.h>

int main(void)
{
    int a[2][3];
    printf("address of a[0][0] is %p\n", &a[0][0]); /* zero */
    printf("address of a[0][2] is %p\n", &a[0][2]); /* should be 4 * (3 * 0 + 2) = 8 */
    printf("address of a[1][2] is %p\n", &a[1][2]); /* should be 4 * (3 * 1 + 2) = 20 */
    return 0;
}
