"C:\Program Files (x86)\pythonxy\swig\swig.exe" -python truncated_normal.i

"C:\Anaconda\Scripts\gcc.bat" -DMS_WIN64 -mdll -O2 -Wall -IC:\Anaconda\include -IC:\Anaconda\Lib\site-packages\numpy\core\include -LC:\Anaconda\libs truncated_normal.c truncated_normal_wrap.c -lpython27 -o _truncated_normal.pyd