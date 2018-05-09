import numpy as np
#0.15582822
if __name__ == "__main__":
    x, y = 1, 2
    p, q = 3, 4
    r = 1

for %R %x in (*.jpg) do (
    for /f "tokens=1-3 delims=. " %%F in ("%%A") do (
       set /a a=%%G
       set zeros=
       if !a! LSS 1000 set zeros=0
       if !a! LSS 100 set zeros=00
       if !a! LSS 10 set zeros=000
       set "name=%%F !zeros!!a!.%%H"
       echo ren "%%A" "!name!"
    )
)
