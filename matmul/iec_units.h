#ifndef __IEC_UNITS_H
#define __IEC_UNITS_H


#define _4KiB       (1<<12)
#define _8KiB       (1<<13)
#define _16KiB      (1<<14)
#define _32KiB      (1<<15)
#define _64KiB      (1<<16)
#define _128KiB     (1<<17)
#define _256KiB     (1<<18)
#define _512KiB     (1<<19)
#define _1MiB       (1<<20)
#define _2MiB       (1<<21)
#define _4MiB       (1<<22)
#define _8MiB       (1<<23)
#define _16MiB      (1<<24)
#define _32MiB      (1<<25)
#define _64MiB      (1<<26)
#define _128MiB     (1<<27)
#define _256MiB     (1<<28)
#define _512MiB     (1<<29)
#define _1GiB       (1<<30)
#define _2GiB       (1<<31)
#define _4GiB       (1<<32)


const char* IEC_SUFFIX[] = {"B", "KiB", "MiB", "GiB", "TiB"};

void format_iec(char* str, const size_t data) {
    float x = data;
    int suffix = 0;
    for (; suffix<5; suffix++) {
        if (x < 8192) {
            break;
        }
        x /= 1024;
    }
    if ((ceil(x) - floor(x)) < 0.01) {
        sprintf(str, "%'d %s", (int) x, IEC_SUFFIX[suffix]);
    } else {
        sprintf(str, "%'.2f %s", x, IEC_SUFFIX[suffix]);
    }
}


#endif // __IEC_UNITS_H
