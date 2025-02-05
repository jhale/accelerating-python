# Session 1: Introduction to Python acceleration tools

:::{note}
This section is given as a presentation.
:::

## Background

### A very quick history of numerical programming

* Humans have to tell computers what to do - *programming*.

* The earliest computers were programmed using punch cards.

:::{figure} babbage.jpg
By Charles Babbage - Upload by Mrjohncummings 2013-08-28 15:10, CC BY-SA 2.0,
https://commons.wikimedia.org/w/index.php?curid=28024313
:::

:::{figure} punchcard.jpg
By Karoly Lorentey - originally posted to Flickr as Punched cards for
programming the Analytical Engine, 1834-71, CC BY 2.0,
https://commons.wikimedia.org/w/index.php?curid=11833648
:::

* And that was the prominent mode of interacting with computers until the
  1960s. 

:::{figure} punchcard2.jpg
By Pete Birkinshaw from Manchester, UK - Used Punchcard, CC BY 2.0,
https://commons.wikimedia.org/w/index.php?curid=49758093
:::

* As computers have become better we have demanded that they do more
  complex tasks.

* The first compilers in the 1950s compile a higher-level programming language
  (typically, Fortran) into assembly and then machine instructions.

```
C AREA OF A TRIANGLE WITH A STANDARD SQUARE ROOT FUNCTION
C INPUT - TAPE READER UNIT 5, INTEGER INPUT
C OUTPUT - LINE PRINTER UNIT 6, REAL OUTPUT
C INPUT ERROR DISPLAY ERROR OUTPUT CODE 1 IN JOB CONTROL LISTING
      READ INPUT TAPE 5, 501, IA, IB, IC
  501 FORMAT (3I5)
C IA, IB, AND IC MAY NOT BE NEGATIVE OR ZERO
C FURTHERMORE, THE SUM OF TWO SIDES OF A TRIANGLE
C MUST BE GREATER THAN THE THIRD SIDE, SO WE CHECK FOR THAT, TOO
      IF (IA) 777, 777, 701
  701 IF (IB) 777, 777, 702
  702 IF (IC) 777, 777, 703
  703 IF (IA+IB-IC) 777, 777, 704
  704 IF (IA+IC-IB) 777, 777, 705
  705 IF (IB+IC-IA) 777, 777, 799
  777 STOP 1
C USING HERON'S FORMULA WE CALCULATE THE
C AREA OF THE TRIANGLE
  799 S = FLOATF (IA + IB + IC) / 2.0
      AREA = SQRTF( S * (S - FLOATF(IA)) * (S - FLOATF(IB)) *
     +     (S - FLOATF(IC)))
      WRITE OUTPUT TAPE 6, 601, IA, IB, IC, AREA
  601 FORMAT (4H A= ,I5,5H  B= ,I5,5H  C= ,I5,8H  AREA= ,F10.2,
     +        13H SQUARE UNITS)
      STOP
      END
```

* With time, compilers became better than humans at writing optimised machine
  instructions.

```
// source
void bar(int a, int b) {
    int x, y;

    x = 555;
    y = a+b;
}

void foo(void) {
    bar(111,222);
}

; compiles to:
bar:
    pushl   %ebp
    movl    %esp, %ebp
    subl    $16, %esp
    movl    $555, -4(%ebp)
    movl    12(%ebp), %eax
    movl    8(%ebp), %edx
    addl    %edx, %eax
    movl    %eax, -8(%ebp)
    leave
    ret

foo:
    pushl   %ebp
    movl    %esp, %ebp
    subl    $8, %esp
    movl    $222, 4(%esp)
    movl    $111, (%esp)
    call    bar
    leave
    ret
```

* Around 2005 intepreted programming languages became increasingly popular for
  writing scientific software, e.g. Python + Numpy, Matlab.

```
C = A*B
```

* Because these languages are often 'good enough', compiled languages no longer
  taught to most scientists - situation unlikely to change.

### Hardware is changing fast

* The modern supercomputer contains not only central processing units, but also
  massively parallel hardware like GPUs and TPUs. 

:::{figure} meluxina.jpg
Meluxina, Bissen, Luxembourg.
:::
  
* AMD MI300x - 153 billion transistors, 20000 stream processors.

:::{figure} amd.jpg
AMD MI300x
:::

* Graphcore - 1400 cores, 8000 threads.

* Apple M-series Neural Engine. 

* Because of their massive parallelism standard programming models cannot take
  advantage of this hardware.

### What's happening today

* A new generation of *just-in-time* compilers and automatic differentiation
  tools for Python code.

* Write the algorithm once, compile, get reasonable performance on CPUs, GPUs
  and possible future hardware.

* Automatic differentiation can produce derivatives of 'arbitrary' computer
  programs - distinct from symbolic differentiation tools.

* I will show on two very simple examples (matrix multiplication and maximum
  likelihood estimate) what is possible.

* I hope to persuade you that these tools are not only easy, but useful -
  saving time, energy and removing some of the frustrating aspects of numerical
  programming. 
