FOR %%f IN (*.c) DO "C:\MinGW32-xy\bin\mingw32-gcc.exe" -c -DBUILD_DLL -O3 -Wall -ansi -pedantic %%f

"C:\MinGW32-xy\bin\mingw32-gcc.exe" -shared -o libtruncated_normal.dll *.o -Wl,--out-implib,libtruncated_normal.a

del *.o