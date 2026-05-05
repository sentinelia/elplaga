# Trabajando con Derivadas — Soluciones Completas

> **Profesora:** Aída Fajardo
> **Tema:** Derivadas — definición, cálculo y aplicaciones (dominio, continuidad, crecimiento, gráficos)

Recordamos las dos formas equivalentes de la **definición de derivada**:

$$
f'(a)=\lim_{x\to a}\frac{f(x)-f(a)}{x-a} \quad=\quad \lim_{h\to 0}\frac{f(a+h)-f(a)}{h}
$$

---

## Ejercicio 1 — Identificar derivadas a partir de límites

Suponiendo que cada límite es un número real, decidimos si corresponde a la derivada de $f$ en algún real.

### a) $\displaystyle\lim_{x\to -3}\frac{f(-3)-f(x)}{-3-x}$

Manipulamos: $-3-x=-(x+3)$ y $f(-3)-f(x)=-(f(x)-f(-3))$. Los signos se cancelan:

$$
\frac{f(-3)-f(x)}{-3-x}=\frac{-(f(x)-f(-3))}{-(x-(-3))}=\frac{f(x)-f(-3)}{x-(-3)}
$$

**→ Es la derivada $f'(-3)$.** ✓

### b) $\displaystyle\lim_{x\to 0}\frac{f(0)-f(x)}{x}$

Reescribimos: $f(0)-f(x)=-(f(x)-f(0))$, denominador $x=x-0$.

$$
\frac{f(0)-f(x)}{x}=-\frac{f(x)-f(0)}{x-0}\;\longrightarrow\;-f'(0)
$$

**→ NO es la derivada de $f$ en ningún real.** Es el **opuesto** de $f'(0)$.

### c) $\displaystyle\lim_{t\to 0}\frac{f(t)}{t}$

$$
\frac{f(t)}{t}=\frac{f(t)-0}{t-0}
$$

Para que el límite sea un número real, necesitamos $f(0)=0$ (de lo contrario el numerador tiende a un número distinto de 0 con denominador tendiendo a 0). Si $f(0)=0$:

$$
\lim_{t\to 0}\frac{f(t)-f(0)}{t-0}=f'(0)
$$

**→ Es $f'(0)$, siempre que $f(0)=0$.** ✓

### d) $\displaystyle\lim_{y\to x}\frac{f(y)-f(x)}{y-x}$

Aquí $x$ es un real fijo y $y$ es la variable que tiende a $x$. Es la definición misma de derivada:

**→ Es $f'(x)$.** ✓

### e) $\displaystyle\lim_{h\to 0}\frac{f(c+h)-f(h)}{h}$

Sumamos y restamos $f(c)$ en el numerador:

$$
\frac{f(c+h)-f(h)}{h}=\frac{f(c+h)-f(c)}{h}+\frac{f(c)-f(h)}{h}
$$

Para que el límite sea real necesitamos $f(c)=f(0)$, y entonces el resultado es $f'(c)-f'(0)$.

**→ NO es la derivada de $f$ en un único real** (es la diferencia $f'(c)-f'(0)$).

---

## Ejercicio 2 — $f(x)=2x^{2}-7x+5$

### a) Derivabilidad por definición

$$
f'(x)=\lim_{h\to 0}\frac{f(x+h)-f(x)}{h}
$$

$$
f(x+h)-f(x)=2(x+h)^2-7(x+h)+5-(2x^2-7x+5)=4xh+2h^2-7h
$$

$$
\frac{f(x+h)-f(x)}{h}=4x+2h-7\;\xrightarrow{h\to 0}\;4x-7
$$

El límite existe y es finito **para todo** $x\in\mathbb{R}$, por lo tanto $f$ es derivable en todo $\mathbb{R}$.

### b) Función derivada

$$
\boxed{f'(x)=4x-7}
$$

### c) Signo de $f'$ y crecimiento

$f'(x)=0\iff x=\dfrac{7}{4}$.

| Intervalo | $f'(x)$ | $f$ |
|---|---|---|
| $\left(-\infty,\,\frac{7}{4}\right)$ | $<0$ | decreciente |
| $\left(\frac{7}{4},\,+\infty\right)$ | $>0$ | creciente |

**Mínimo absoluto** en $x=\dfrac{7}{4}$, con $f\!\left(\dfrac{7}{4}\right)=-\dfrac{9}{8}$.

### d) Límites y gráfico

- $\displaystyle\lim_{x\to\pm\infty}f(x)=+\infty$
- Raíces: $2x^2-7x+5=0\Rightarrow x=1$ ó $x=\dfrac{5}{2}$
- Ordenada al origen: $f(0)=5$

Es una **parábola** con concavidad hacia arriba, vértice $\left(\frac{7}{4},-\frac{9}{8}\right)$, cortes en $x=1,\, x=\frac{5}{2}$.

```
        y
        |
      5 +•                        •
        |  \                    /
        |   \                  /
        |    \  •___________•/
       -+-----+--+--+--+--+--+--→ x
        |     1  7/4  5/2
              vértice (7/4, -9/8)
```

---

## Ejercicio 3 — $f(x)=\dfrac{2}{x-3}+x$

### a) Derivabilidad por definición

Dominio: $\mathbb{R}\setminus\{3\}$.

$$
f(x+h)-f(x)=\frac{2}{x+h-3}-\frac{2}{x-3}+h=\frac{-2h}{(x+h-3)(x-3)}+h
$$

$$
\frac{f(x+h)-f(x)}{h}=\frac{-2}{(x+h-3)(x-3)}+1\;\xrightarrow{h\to 0}\;1-\frac{2}{(x-3)^2}
$$

El límite existe en todo el dominio, por lo tanto $f$ es derivable en $\mathbb{R}\setminus\{3\}$.

### b) Función derivada

$$
\boxed{f'(x)=1-\frac{2}{(x-3)^2}=\frac{(x-3)^2-2}{(x-3)^2}}
$$

### c) Signo de $f'$ y crecimiento

$(x-3)^2>0$ siempre (en el dominio). $f'(x)=0\iff (x-3)^2=2\iff x=3\pm\sqrt{2}$.

| Intervalo | $f'$ | $f$ |
|---|---|---|
| $(-\infty,\,3-\sqrt{2})$ | $>0$ | creciente |
| $(3-\sqrt{2},\,3)$ | $<0$ | decreciente |
| $(3,\,3+\sqrt{2})$ | $<0$ | decreciente |
| $(3+\sqrt{2},\,+\infty)$ | $>0$ | creciente |

- **Máximo local** en $x=3-\sqrt{2}$: $f(3-\sqrt{2})=3-2\sqrt{2}\approx 0{,}17$
- **Mínimo local** en $x=3+\sqrt{2}$: $f(3+\sqrt{2})=3+2\sqrt{2}\approx 5{,}83$

### d) Límites y bosquejo

- $\displaystyle\lim_{x\to 3^-}f(x)=-\infty,\quad \lim_{x\to 3^+}f(x)=+\infty$ (asíntota vertical $x=3$)
- $\displaystyle\lim_{x\to\pm\infty}\bigl(f(x)-x\bigr)=\lim_{x\to\pm\infty}\frac{2}{x-3}=0$ → **asíntota oblicua $y=x$**

El gráfico tiene dos ramas separadas por $x=3$, con asíntota oblicua $y=x$ y los extremos locales descritos.

---

## Ejercicio 4 — $f(x)=-x^{3}+5x^{2}+13x+7$

### a) Derivabilidad por definición

$$
f(x+h)-f(x)=h\bigl(-3x^{2}-3xh-h^{2}+10x+5h+13\bigr)
$$

$$
\frac{f(x+h)-f(x)}{h}\xrightarrow{h\to 0}-3x^{2}+10x+13
$$

Existe para todo $x\in\mathbb{R}$ ⇒ $f$ derivable en todo $\mathbb{R}$.

### b) Función derivada

$$
\boxed{f'(x)=-3x^{2}+10x+13=-3(x+1)\!\left(x-\tfrac{13}{3}\right)}
$$

### c) Signo de $f'$ y crecimiento

Raíces de $f'$: $x=-1$ y $x=\dfrac{13}{3}$. Como $f'$ es una parábola hacia abajo:

| Intervalo | $f'$ | $f$ |
|---|---|---|
| $(-\infty,-1)$ | $<0$ | decreciente |
| $\left(-1,\,\frac{13}{3}\right)$ | $>0$ | creciente |
| $\left(\frac{13}{3},+\infty\right)$ | $<0$ | decreciente |

- **Mínimo local** en $x=-1$: $f(-1)=1+5-13+7=0$
- **Máximo local** en $x=\frac{13}{3}$: $f\!\left(\frac{13}{3}\right)=\dfrac{2048}{27}\approx 75{,}85$

### d) Límites y bosquejo

- $\displaystyle\lim_{x\to-\infty}f(x)=+\infty,\qquad \lim_{x\to+\infty}f(x)=-\infty$
- Factorización: $f(x)=-(x+1)^{2}(x-7)$ → raíces $x=-1$ (doble, donde toca al eje) y $x=7$.

Curva cúbica con $a<0$: viene de $+\infty$, decrece hasta $(-1,0)$, crece hasta $\left(\frac{13}{3},\frac{2048}{27}\right)$ y luego decrece a $-\infty$, cortando al eje en $x=7$.

---

## Ejercicio 5 — Reglas de derivación

| # | $f(x)$ | $f'(x)$ |
|---|---|---|
| 1 | $3x^{3}-7x+4$ | $9x^{2}-7$ |
| 2 | $4e^{x}-3\ln x+5$ | $4e^{x}-\dfrac{3}{x}$ |
| 3 | $4\sqrt{x}-6$ | $\dfrac{2}{\sqrt{x}}$ |
| 4 | $(x^{2}+3x+1)e^{x}$ | $(x^{2}+5x+4)e^{x}=(x+1)(x+4)e^{x}$ |
| 5 | $x\ln x$ | $\ln x+1$ |
| 6 | $(x+1)\sqrt{x}$ | $\dfrac{3x+1}{2\sqrt{x}}$ |
| 7 | $2(x-1)e^{x}$ | $2x\,e^{x}$ |
| 8 | $\dfrac{3x+2}{x+4}$ | $\dfrac{10}{(x+4)^{2}}$ |
| 9 | $\dfrac{x}{e^{x}}$ | $\dfrac{1-x}{e^{x}}$ |
| 10 | $\dfrac{2e^{x}}{3x^{2}-1}$ | $\dfrac{2e^{x}(3x^{2}-6x-1)}{(3x^{2}-1)^{2}}$ |
| 11 | $\dfrac{x+1}{\sqrt{x}}$ | $\dfrac{x-1}{2x\sqrt{x}}$ |
| 12 | $\dfrac{xe^{x}}{x+1}$ | $\dfrac{e^{x}(x^{2}+x+1)}{(x+1)^{2}}$ |
| 13 | $\dfrac{(x+1)e^{x}}{x^{2}}$ | $\dfrac{e^{x}(x^{2}-2)}{x^{3}}$ |
| 14 | $\ln(2x+1)$ | $\dfrac{2}{2x+1}$ |
| 15 | $e^{1/x}$ | $-\dfrac{e^{1/x}}{x^{2}}$ |
| 16 | $\ln\left|\dfrac{x+1}{x-1}\right|$ | $-\dfrac{2}{x^{2}-1}$ |

### Observaciones para los detallados:

**5.4** Producto: $(2x+3)e^{x}+(x^{2}+3x+1)e^{x}$.

**5.7** $2e^{x}+2(x-1)e^{x}=2e^{x}(1+x-1)=2xe^{x}$.

**5.16** $\ln\left|\frac{x+1}{x-1}\right|=\ln|x+1|-\ln|x-1|$, derivando: $\frac{1}{x+1}-\frac{1}{x-1}=\frac{-2}{x^{2}-1}$.

### Ejercicio 5.17

**a)** $f(x)=(x+2)e^{1/x}$:

$$
f'(x)=e^{1/x}+(x+2)e^{1/x}\!\left(-\frac{1}{x^{2}}\right)=e^{1/x}\frac{x^{2}-x-2}{x^{2}}=\boxed{\dfrac{(x-2)(x+1)e^{1/x}}{x^{2}}}
$$

**b)** $g(x)=\dfrac{(x^{2}-x-2)e^{1/x}}{x^{2}}$. Notar que $g(x)=f'(x)$. Derivando con regla del producto/cociente:

$$
g'(x)=\boxed{\dfrac{(5x+2)\,e^{1/x}}{x^{4}}}
$$

### Ejercicio 5.18

**a)** $f(x)=(x-1)e^{x/(x+2)}$. Como $\left(\dfrac{x}{x+2}\right)'=\dfrac{2}{(x+2)^{2}}$:

$$
f'(x)=e^{x/(x+2)}+(x-1)e^{x/(x+2)}\cdot\frac{2}{(x+2)^{2}}=e^{x/(x+2)}\cdot\frac{(x+2)^{2}+2(x-1)}{(x+2)^{2}}
$$

$$
\boxed{f'(x)=\dfrac{(x^{2}+6x+2)\,e^{x/(x+2)}}{(x+2)^{2}}}
$$

**b)** $g(x)=\dfrac{(x^{2}+6x+2)\,e^{x/(x+2)}}{(x+2)^{2}}=f'(x)$. Su derivada es $f''(x)$:

$$
\boxed{g'(x)=\dfrac{4(4x+5)\,e^{x/(x+2)}}{(x+2)^{4}}}
$$

---

## Ejercicio 6 — Derivabilidad por definición

Estudiamos por definición allí donde las reglas no aseguran derivabilidad: ceros de un valor absoluto (posible "pico") o ceros bajo una raíz (posible tangente vertical).

### 6.1) $f(x)=\dfrac{1}{9}\,\bigl|(x+3)^{2}(x-1)\bigr|$

El argumento $(x+3)^{2}(x-1)$ se anula en $x=-3$ (raíz **doble**) y en $x=1$ (raíz **simple**).

**En $x=-3$:** la raíz es doble, así que $(x+3)^{2}(x-1)\le 0$ a ambos lados de $-3$ (porque $(x+3)^{2}\ge 0$ y $x-1<0$). El valor absoluto coincide con $-(x+3)^{2}(x-1)$ en un entorno: la función es polinómica en torno de $-3$, **derivable**.

Por definición:
$$
\frac{f(x)-f(-3)}{x+3}=\frac{1}{9}(x+3)(1-x)\xrightarrow{x\to-3}0
$$
Luego $f'(-3)=0$.

**En $x=1$:** raíz simple, el signo del argumento cambia. Calculamos derivadas laterales:

$$
\lim_{x\to 1^{+}}\frac{f(x)}{x-1}=\lim_{x\to 1^{+}}\frac{(x+3)^{2}(x-1)}{9(x-1)}=\frac{16}{9}
$$

$$
\lim_{x\to 1^{-}}\frac{f(x)}{x-1}=\lim_{x\to 1^{-}}\frac{-(x+3)^{2}(x-1)}{9(x-1)}=-\frac{16}{9}
$$

**No coinciden ⇒ $f$ no es derivable en $x=1$** (hay un pico).

**Función derivada:**
$$
f'(x)=\begin{cases}\dfrac{(x+3)(3x+1)}{9} & x>1\\[6pt]-\dfrac{(x+3)(3x+1)}{9} & x<1\end{cases}
$$

### 6.2) $f(x)=\sqrt{|x^{2}+4x+3|}=\sqrt{|(x+1)(x+3)|}$

El argumento se anula en $x=-3$ y $x=-1$. La raíz cuadrada puede dar tangente vertical.

**En $x=-3$:** Tomando $x=-3+h$ para $h>0$ (zona donde el polinomio es $\le 0$):
$$
\frac{f(x)}{x+3}=\frac{\sqrt{(2-h)h}}{h}\sim\frac{\sqrt{2h}}{h}=\sqrt{\frac{2}{h}}\to+\infty
$$
y para $h<0$ (zona donde el polinomio es $\ge 0$): $\to -\infty$.

**$f$ no es derivable en $x=-3$** (cúspide / tangente vertical).

**En $x=-1$:** análogo por simetría, **no es derivable**.

**Función derivada (donde existe):**
$$
f'(x)=\begin{cases}\dfrac{x+2}{\sqrt{x^{2}+4x+3}} & x<-3\,\text{ ó }\,x>-1\\[6pt] -\dfrac{x+2}{\sqrt{-(x^{2}+4x+3)}} & -3<x<-1\end{cases}
$$

### 6.3) $f(x)=\bigl(\sqrt{|x+1|}\bigr)^{3}=|x+1|^{3/2}$

El único punto problemático es $x=-1$.

$$
\frac{f(x)-f(-1)}{x+1}=\frac{|x+1|^{3/2}}{x+1}=\operatorname{sg}(x+1)\cdot |x+1|^{1/2}\xrightarrow{x\to-1}0
$$

Ambas derivadas laterales valen $0$ ⇒ **$f$ es derivable en $x=-1$ con $f'(-1)=0$**.

**Función derivada:**
$$
f'(x)=\frac{3}{2}\,\operatorname{sg}(x+1)\sqrt{|x+1|}=\begin{cases}\frac{3}{2}\sqrt{x+1} & x\ge-1\\[4pt]-\frac{3}{2}\sqrt{-x-1} & x\le-1\end{cases}
$$

---

## Ejercicio 7 — Estudio completo

### a) $f(x)=x-2+\ln|x+1|$

**Dominio:** $\mathbb{R}\setminus\{-1\}$.

**Continuidad:** continua en su dominio (suma de funciones continuas).

**Límites:**
- $\displaystyle\lim_{x\to-\infty}f(x)=-\infty$ (admitido).
- $\displaystyle\lim_{x\to+\infty}f(x)=+\infty$.
- $\displaystyle\lim_{x\to-1}f(x)=-\infty$ (porque $\ln|x+1|\to-\infty$). **Asíntota vertical $x=-1$**.

**Derivada:**
$$
f'(x)=1+\frac{1}{x+1}=\frac{x+2}{x+1}
$$

| Intervalo | $f'$ | $f$ |
|---|---|---|
| $(-\infty,-2)$ | $>0$ | creciente |
| $(-2,-1)$ | $<0$ | decreciente |
| $(-1,+\infty)$ | $>0$ | creciente |

**Máximo local** en $x=-2$: $f(-2)=-2-2+\ln 1=-4$.

**Bosquejo:**
- En $(-\infty,-2)$ crece desde $-\infty$ hasta $-4$.
- En $(-2,-1)$ decrece desde $-4$ hasta $-\infty$.
- En $(-1,+\infty)$ crece desde $-\infty$ hasta $+\infty$, cortando al eje $x$ entre $1$ y $2$.

### b) $f(x)=(x+6)\,e^{1/x}$

**Dominio:** $\mathbb{R}\setminus\{0\}$.

**Continuidad:** continua en su dominio.

**Límites:**
- $\displaystyle\lim_{x\to-\infty}f(x)=-\infty$ (porque $e^{1/x}\to 1$).
- $\displaystyle\lim_{x\to+\infty}f(x)=+\infty$.
- $\displaystyle\lim_{x\to 0^{-}}f(x)=6\cdot 0=0$ (porque $1/x\to-\infty\Rightarrow e^{1/x}\to 0$).
- $\displaystyle\lim_{x\to 0^{+}}f(x)=+\infty$ ($1/x\to+\infty\Rightarrow e^{1/x}\to+\infty$). **Asíntota vertical (lateral derecha) $x=0$**.

**Asíntota oblicua en $\pm\infty$:** $m=\lim\frac{f(x)}{x}=1$, $b=\lim(f(x)-x)=\lim\bigl(x(e^{1/x}-1)+6e^{1/x}\bigr)=1+6=7$. **Asíntota $y=x+7$**.

**Derivada:**
$$
f'(x)=e^{1/x}+(x+6)e^{1/x}\!\left(-\frac{1}{x^{2}}\right)=\frac{e^{1/x}(x^{2}-x-6)}{x^{2}}=\frac{e^{1/x}(x-3)(x+2)}{x^{2}}
$$

Como $e^{1/x}>0$ y $x^{2}>0$, el signo lo da $(x-3)(x+2)$:

| Intervalo | $f'$ | $f$ |
|---|---|---|
| $(-\infty,-2)$ | $>0$ | creciente |
| $(-2,0)$ | $<0$ | decreciente |
| $(0,3)$ | $<0$ | decreciente |
| $(3,+\infty)$ | $>0$ | creciente |

- **Máximo local** en $x=-2$: $f(-2)=4\,e^{-1/2}=\dfrac{4}{\sqrt{e}}\approx 2{,}43$.
- **Mínimo local** en $x=3$: $f(3)=9\,e^{1/3}\approx 12{,}56$.

### c) $f(x)=x-2+\ln\left|\dfrac{x^{2}}{x+3}\right|$

**Dominio:** $x\neq 0$ (porque $x^{2}/|x+3|=0\Rightarrow\ln$ no definido) y $x\neq-3$. Es decir, $\mathbb{R}\setminus\{-3,0\}$.

Reescribimos: $\ln\left|\dfrac{x^{2}}{x+3}\right|=2\ln|x|-\ln|x+3|$.

**Continuidad:** continua en su dominio.

**Límites:**
- $\displaystyle\lim_{x\to-\infty}f(x)=-\infty$ (admitido).
- $\displaystyle\lim_{x\to+\infty}f(x)=+\infty$.
- $\displaystyle\lim_{x\to-3}f(x)=+\infty$ (porque $|x+3|\to 0\Rightarrow -\ln|x+3|\to+\infty$). **Asíntota vertical $x=-3$**.
- $\displaystyle\lim_{x\to 0}f(x)=-\infty$ (porque $2\ln|x|\to-\infty$). **Asíntota vertical $x=0$**.

**Derivada:**
$$
f'(x)=1+\frac{2}{x}-\frac{1}{x+3}=\frac{x(x+3)+2(x+3)-x}{x(x+3)}=\frac{x^{2}+4x+6}{x(x+3)}
$$

El numerador $x^{2}+4x+6$ tiene discriminante $16-24=-8<0$, así que **siempre es positivo**. Entonces el signo de $f'$ depende solamente de $x(x+3)$:

| Intervalo | $x(x+3)$ | $f'$ | $f$ |
|---|---|---|---|
| $(-\infty,-3)$ | $>0$ | $>0$ | creciente |
| $(-3,0)$ | $<0$ | $<0$ | decreciente |
| $(0,+\infty)$ | $>0$ | $>0$ | creciente |

**No hay extremos locales** (la derivada nunca se anula).

**Bosquejo:**
- En $(-\infty,-3)$ crece desde $-\infty$ hasta $+\infty$.
- En $(-3,0)$ decrece desde $+\infty$ hasta $-\infty$.
- En $(0,+\infty)$ crece desde $-\infty$ hasta $+\infty$.

---

## Apéndice — Tabla de derivadas elementales utilizadas

| Función | Derivada |
|---|---|
| $x^{n}$ | $nx^{n-1}$ |
| $e^{x}$ | $e^{x}$ |
| $\ln|x|$ | $1/x$ |
| $\sqrt{x}$ | $1/(2\sqrt{x})$ |
| $f(x)\cdot g(x)$ | $f'g+fg'$ |
| $f(x)/g(x)$ | $(f'g-fg')/g^{2}$ |
| $f(g(x))$ | $f'(g(x))\cdot g'(x)$ |
