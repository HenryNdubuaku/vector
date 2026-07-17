# Vector by example

Generated from the test suite by `cargo test` — every program and its output are verified on each run. Do not edit by hand.

## algos

```python
x = [3.0, 1.0, 2.0]
print(sort(x))
print(argsort(x))
print(argmax(x))
print(argmin(x))
print(one_hot(1.0, 3))
print(one_hot([2.0, 0.0], 3))
print(take([10.0, 20.0, 30.0], [2.0, 0.0]))
print(cumsum([1.0, 2.0, 3.0]))
print(softmax([0.0, 0.0]))
print(relu([-1.0, 2.0]))
print(sigmoid(0.0))
print(std([1.0, 3.0]))
print(norm([3.0, 4.0]))
print(logsumexp([0.0, 0.0]))
print(where([1.0, 2.0] == [1.0, 3.0], 1.0, 0.0))
print(where([1.0, 2.0] != [1.0, 3.0], 1.0, 0.0))

m = [[3.0, 1.0], [0.0, 5.0]]
fn am(r):
  argmax(r)

print(vmap(am, m))

fn srt(r):
  sort(r)

print(vmap(srt, m))

fn f(v):
  sum(cumsum(v))

print(grad(f, [1.0, 2.0, 3.0]))

fn g(v):
  sum(take(v, [2.0, 2.0]))

print(grad(g, [1.0, 2.0, 3.0]))
```

Output:

```
[1, 2, 3] : f32
[1, 2, 0] : f32
0 : f32
1 : f32
[0, 1, 0] : f32
[[0, 0, 1], [1, 0, 0]] : f32
[30, 10] : f32
[1, 3, 6] : f32
[0.5, 0.5] : f32
[0, 2] : f32
0.5 : f32
1 : f32
5 : f32
0.6931472 : f32
[1, 0] : f32
[0, 1] : f32
[0, 1] : f32
[[1, 3], [0, 5]] : f32
[3, 2, 1] : f32
[0, 0, 2] : f32
```

## arithmetic

```python
# comments run to end of line
print(-3.0 + 2.0 * 5.0)  # including after expressions
print(-(1.0 + 2.0))
print(2.0 / -4.0)
print(1.0 - -1.0)
print(--5.0)
```

Output:

```
7 : f32
-3 : f32
-0.5 : f32
2 : f32
5 : f32
```

## broadcast

```python
m = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
b = [10.0, 20.0, 30.0]
print(m + b)
print(m * 2.0)
```

Output:

```
[[11, 22, 33], [14, 25, 36]] : f32
[[2, 4, 6], [8, 10, 12]] : f32
```

## dtypes

```python
xs = f32([3.0, 4.0])
print(xs + 0.5)
print(f64(xs))
```

Output:

```
[3.5, 4.5] : f32
[3, 4] : f64
```

## for_where

```python
w = 1.0
for i in 0..4:
  w = w * 2.0
print(w)

acc = 0.0
for i in 0..5:
  acc = acc + i
print(acc)

fn sq_loss(v):
  d = v - 3.0
  sum(d * d)

v = 0.0
for i in 0..20:
  v = v - 0.5 * grad(sq_loss, v)
print(v)

x = [-1.0, 0.5, 2.0]
print(where(x > 0.0, x, 0.0))
print(where(x < 1.0, 1.0, -1.0))

fn hinge_like(h):
  sum(where(h > 0.0, h, h * 0.0))

print(grad(hinge_like, [2.0, -3.0]))
```

Output:

```
16 : f32
10 : f32
3 : f32
[0, 0.5, 2] : f32
[1, 1, -1] : f32
[1, 0] : f32
```

## functions

```python
fn norm_sq(xs):
  squares = xs * xs
  sum(squares)

xs = f32([3.0, 4.0])
print(norm_sq(xs))
```

Output:

```
25 : f32
```

## gather

```python
e = reshape(arange(32), 8, 4)
print(take(e, [2.0, 0.0, 2.0]))
print(take(e, 3.0))
print(take([10.0, 20.0, 30.0], 1.0))

fn f(w):
  sum(take(w, [1.0, 1.0, 3.0]))

print(grad(f, reshape(arange(32), 8, 4)))

fn s(v):
  sum(sort(v) * [1.0, 2.0, 3.0])

print(grad(s, [3.0, 1.0, 2.0]))

fn pick(r):
  take(r, [1.0, 0.0])

print(vmap(pick, [[10.0, 20.0], [30.0, 40.0]]))
```

Output:

```
[[8, 9, 10, 11], [0, 1, 2, 3], [8, 9, 10, 11]] : f32
[12, 13, 14, 15] : f32
20 : f32
[[0, 0, 0, 0], [2, 2, 2, 2], [0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]] : f32
[3, 1, 2] : f32
[[20, 10], [40, 30]] : f32
```

## generate

```python
print(arange(5))
print(arange(2, 5))
print(arange(1.0, 2.0, 0.25))
print(arange(3, 0, -1))
print(reshape(arange(6), 2, 3))
print(sin(0.0))
print(cos(0.0))

fn s(x):
  sum(sin(x))

print(grad(s, [0.0, 0.0]))

fn r(x):
  sum(reshape(x, 4) * [1.0, 2.0, 3.0, 4.0])

print(grad(r, [[1.0, 1.0], [1.0, 1.0]]))
print(linspace(0.0, 1.0, 5))
half = 0.5
print(linspace(-half, half, 3))
n = 4.0
print(arange(n))
print(linspace(-pi, pi, 3))
print(sin(pi / 2.0))
```

Output:

```
[0, 1, 2, 3, 4] : f32
[2, 3, 4] : f32
[1, 1.25, 1.5, 1.75] : f32
[3, 2, 1] : f32
[[0, 1, 2], [3, 4, 5]] : f32
0 : f32
1 : f32
[1, 1] : f32
[[1, 2], [3, 4]] : f32
[0, 0.25, 0.5, 0.75, 1] : f32
[-0.5, 0, 0.5] : f32
[0, 1, 2, 3] : f32
[-3.1415927, 0, 3.1415927] : f32
1 : f32
```

## grad

```python
fn norm_sq(xs):
  sum(xs * xs)

print(grad(norm_sq, [3.0, 4.0]))

fn hinge(x):
  sum(maximum(x, 0.0))

print(grad(hinge, [1.0, -2.0]))

fn mean_sq(x):
  mean(x * x)

print(grad(mean_sq, [3.0, 6.0]))

fn lin(w):
  sum(matmul([[1.0, 2.0], [3.0, 4.0]], w))

print(grad(lin, [[1.0, 1.0], [1.0, 1.0]]))

fn chain(x):
  sum(sqrt(exp(x)))

print(grad(chain, [0.0]))

fn recip(x):
  sum(1.0 / x)

print(grad(recip, [2.0]))

fn biased(b):
  sum([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]] + b)

print(grad(biased, [0.0, 0.0, 0.0]))

fn stacked(x):
  sum([x, x] * [2.0, 3.0])

print(grad(stacked, 5.0))

fn first(w, b):
  sum(w * b)

print(grad(first, [1.0, 2.0], [3.0, 4.0]))

fn times_five(x):
  sum(x * 5.0)

print(grad(times_five, [5.0]))
```

Output:

```
[6, 8] : f32
[1, 0] : f32
[3, 6] : f32
[[4, 4], [6, 6]] : f32
[0.5] : f32
[-0.25] : f32
[2, 2, 2] : f32
5 : f32
[3, 4] : f32
[5] : f32
```

## grad2

```python
fn sq(x):
  sum(x * x)

fn dsq(x):
  grad(sq, x)

print(grad(dsq, 3.0))

fn stacked(x):
  sum([x, x] * [2.0, 3.0])

fn dstacked(x):
  grad(stacked, x)

print(grad(dstacked, 5.0))

fn cube(x):
  sum(x * x * x)

fn dcube(x):
  grad(cube, x)

fn ddcube(x):
  grad(dcube, x)

print(grad(dcube, 2.0))
print(grad(ddcube, 2.0))

fn growth(x):
  sum(exp(x))

fn dgrowth(x):
  grad(growth, x)

print(grad(dgrowth, 0.0))
```

Output:

```
2 : f32
0 : f32
12 : f32
6 : f32
1 : f32
```

## jacobian

```python
fn poly(x):
  x * x * 2.0

print(jacobian(poly, [1.0, 2.0]))

fn affine(x):
  matmul([[1.0, 2.0], [3.0, 4.0]], x)

print(jacobian(affine, [5.0, 7.0]))

fn jpoly(x):
  jacobian(poly, x)

print(vmap(jpoly, [[1.0, 2.0], [3.0, 1.0]]))
```

Output:

```
[[4, 0], [0, 8]] : f32
[[1, 2], [3, 4]] : f32
[[[4, 0], [0, 8]], [[12, 0], [0, 4]]] : f32
```

## linear

```python
layer = Linear(2, 3)
print(layer.b)
print(layer([0.0, 0.0]))
g = glorot_uniform(8, 8)
print(sum(where(g * g <= 0.375, 1.0, 0.0)))
h = he_normal(4, 4)
print(sum(h * 0.0))
print(sum(lecun_uniform(6, 2) * 0.0))
```

Output:

```
[0, 0, 0] : f32
[0, 0, 0] : f32
64 : f32
0 : f32
0 : f32
```

## load

```python
m = load("tests/cases/data/m.npy")
v = load("tests/cases/data/v.npy")
print(m + 0.5)
print(sum(v * v))
print(matmul(m, f32([1.0, 1.0])))

fn q(x):
  sum(x * x)

print(grad(q, v))
```

Output:

```
[[2, 3], [4, 5]] : f32
25 : f64
[4, 8] : f32
[6, 8] : f64
```

## loop_print

```python
w = 1.0
for i in 0..3:
  w = w * 2.0
  print(w)
print(w)

acc = 0.0
for x in 0.0..1.0 by 0.5:
  acc = acc + x
  print({a: acc, b: x})
print(acc)

fn f(v):
  y = v
  for k in 0..2:
    y = y * y
    print(sum(y))
  sum(y)

print(grad(f, [2.0]))
```

Output:

```
i 0: 2 : f32
i 1: 4 : f32
i 2: 8 : f32
8 : f32
x 0: a: 0 : f32
x 0.5: a: 0.5 : f32
x 0: b: 0 : f32
x 0.5: b: 0.5 : f32
0.5 : f32
k 0: 4 : f32
k 1: 16 : f32
[32] : f32
```

## loops

```python
fn looped(x):
  y = x
  for i in 0..3:
    y = y * y
  sum(y)

print(grad(looped, [2.0]))

fn inner_double(x):
  y = x
  for j in 0..2:
    y = y * 2.0
  y

acc = 1.0
for i in 0..3:
  acc = inner_double(acc)
print(acc)

state = {v: [1.0, 2.0], scale: 2.0}
for i in 0..3:
  state = state * state.scale / 2.0 + {v: [1.0, 1.0], scale: 1.0} * 0.0 + state * 0.0 + state
print(state.v)
acc = 0.0
for i in 0..10 by 2:
  acc = acc + i
print(acc)

for i in 5..0 by -1:
  acc = acc + i
print(acc)

for x in 0.0..1.0 by 0.25:
  acc = acc + x
print(acc)

fn f(v):
  y = v
  for i in 0..4 by 2:
    y = y * 2.0
  sum(y)

print(grad(f, [3.0]))
```

Output:

```
[1024] : f32
64 : f32
[42, 84] : f32
20 : f32
35 : f32
36.5 : f32
[4] : f32
```

## math

```python
print(exp(0.0))
print(sqrt([4.0, 9.0]))
print(tanh(0.0))
print(log(1.0))
print(maximum(-1.0, [0.5, -2.0]))
print(minimum(2.0, 1.0))
```

Output:

```
1 : f32
[2, 3] : f32
0 : f32
0 : f32
[0.5, -1] : f32
1 : f32
```

## matmul

```python
a = [[1.0, 2.0], [3.0, 4.0]]
b = [[5.0, 6.0], [7.0, 8.0]]
print(matmul(a, b))
v = [1.0, 1.0]
print(matmul(a, v))
print(matmul(v, a))
print(matmul(v, v))
```

Output:

```
[[19, 22], [43, 50]] : f32
[3, 7] : f32
[4, 6] : f32
2 : f32
```

## modules

```python
module Scale(k):
  s = 0.0 + k
  forward(self, x):
    self.s * x

sc = Scale(3)
print(sc([1.0, 2.0]))
print(sc.s)

fn l(m):
  sum(m([2.0]))

print(grad(l, sc))

sc2 = sc - 0.5 * grad(l, sc)
print(sc2.s)
print(sc2([1.0]))

module Z(n):
  v = zeros(n)
  forward(self, x):
    self.v + x

z = Z(3)
print(z(1.0))

module Pair():
  a = Scale(2)
  b = Scale(4)
  forward(self, x):
    self.b(self.a(x))

p = Pair()
print(p([1.0]))

fn pl(m):
  sum(m([1.0]))

print(grad(pl, p))
print(sum(randn(2, 2) * 0.0))

module Quad(k):
  c = 0.0 + k
  forward(self, x):
    self.c * x
  energy(self):
    e = self.c - 3.0
    sum(e * e)
  scaled_energy(self, w):
    w * self.energy()

q = Quad(1)
print(q.energy())
print(grad(q.energy))
q2 = q - 0.25 * grad(q.energy)
print(q2.c)
print(q2.energy())
print(q2([2.0]))
print(q.scaled_energy(2.0))
print(grad(q.scaled_energy, 2.0))
```

Output:

```
[3, 6] : f32
3 : f32
s: 2 : f32
2 : f32
[2] : f32
[1, 1, 1] : f32
[8] : f32
a.s: 4 : f32
b.s: 2 : f32
0 : f32
4 : f32
c: -4 : f32
2 : f32
1 : f32
[4] : f32
8 : f32
c: -8 : f32
```

## optimizer

```python
fn f(x):
  sum(x * x)

state = {x: [1.0], m: [0.0]}
for i in 0..3:
  g = grad(f, state.x)
  m = 0.5 * state.m + g
  state = {x: state.x - 0.5 * m, m: m}
print(state.x)
print(state.m)
```

Output:

```
[-0.25] : f32
[-0.5] : f32
```

## random

```python
x = zeros(4096) + 1.0
m = mean(dropout(x, 0.5))
print(where(m > 0.85, 1.0, 0.0) * where(m < 1.15, 1.0, 0.0))

u = uniform(4096)
print(where(mean(u) > 0.4, 1.0, 0.0) * where(mean(u) < 0.6, 1.0, 0.0))
print(where(max(u) <= 1.0, 1.0, 0.0) * where(min(u) >= 0.0, 1.0, 0.0))

fn pick(r):
  sample(r * 1.0)

logits = zeros(1000, 3) + [0.0, 10.0, 0.0]
hits = mean(where(vmap(pick, logits) == 1.0, 1.0, 0.0))
print(where(hits > 0.95, 1.0, 0.0))

prev = zeros(16)
repeats = 0.0
for i in 0..8:
  u2 = uniform(16)
  repeats = repeats + sum(where(u2 == prev, 1.0, 0.0))
  prev = u2 * 1.0
print(repeats)

p = permutation(64)
print(sum(where(sort(p) == arange(64), 1.0, 0.0)))
print(where(sum(where(p == arange(64), 1.0, 0.0)) < 64.0, 1.0, 0.0))

fn dloss(w):
  mean(dropout(w, 0.25) * w)

g = grad(dloss, zeros(64) + 2.0)
print(where(sum(g * g) > 0.0, 1.0, 0.0))
```

Output:

```
1 : f32
1 : f32
1 : f32
1 : f32
0 : f32
64 : f32
1 : f32
1 : f32
```

## records

```python
p = {a: [1.0, 2.0], b: 3.0}
print(p)
print(p.a + p.b)
q = p * 2.0
print(q.a)
r = p + q
print(r.b)

fn nsq(p):
  sum(p.a * p.a) + p.b * p.b

print(grad(nsq, p))

nested = {inner: {v: [1.0, 1.0]}, w: 2.0}
print(nested.inner.v)

fn nloss(n):
  sum(n.inner.v * n.inner.v) * n.w

print(grad(nloss, nested))

s = {x: 1.0}
for i in 0..3:
  s = s + s
print(s.x)
```

Output:

```
a: [1, 2] : f32
b: 3 : f32
[4, 5] : f32
[2, 4] : f32
9 : f32
a: [2, 4] : f32
b: 6 : f32
[1, 1] : f32
inner.v: [4, 4] : f32
w: 2 : f32
8 : f32
```

## reduce

```python
m = [[1.0, 2.0], [3.0, 4.0]]
print(sum(m))
print(sum(m, 0))
print(sum(m, 1))
print(mean(m))
print(mean(m, 1))
```

Output:

```
10 : f32
[4, 6] : f32
[3, 7] : f32
2.5 : f32
[1.5, 3.5] : f32
```

## reduce_minmax

```python
m = [[1.0, 5.0], [7.0, 3.0]]
print(max(m))
print(max(m, 0))
print(max(m, 1))
print(min(m))
print(maximum([1.0, 4.0], [3.0, 2.0]))
print(minimum(2.0, [1.0, 3.0]))

fn peak(x):
  max(x)

print(grad(peak, [1.0, 5.0, 3.0]))

fn rowpeaks(x):
  sum(max(x, 1))

print(grad(rowpeaks, [[1.0, 5.0], [7.0, 3.0]]))
```

Output:

```
7 : f32
[7, 5] : f32
[5, 7] : f32
1 : f32
[3, 4] : f32
[1, 2] : f32
[0, 1, 0] : f32
[[0, 1], [1, 0]] : f32
```

## resize

```python
x = [[0.0, 1.0], [1.0, 0.0]]
print(resize(x, 4, 4))

fn f(v):
  sum(resize(v, 4, 4))

print(grad(f, [[0.0, 1.0], [1.0, 0.0]]))

m = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
print(crop(m, 1, 0, 2, 2))

fn c(v):
  sum(crop(v, 0, 1, 2, 2))

print(grad(c, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
```

Output:

```
[[0, 0.25, 0.75, 1], [0.25, 0.375, 0.625, 0.75], [0.75, 0.625, 0.375, 0.25], [1, 0.75, 0.25, 0]] : f32
[[4, 4], [4, 4]] : f32
[[4, 5], [7, 8]] : f32
[[0, 1, 1], [0, 1, 1]] : f32
```

## slicing

```python
d = [10.0, 20.0, 30.0, 40.0]
print(slice(d, 1.0, 2))
print(slice(d, 0.0, 3))
m = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
print(slice(m, 2.0, 1))
print(floor([1.7, -1.2, 3.0]))

fn pick(x):
  sum(slice(x, 1.0, 2))

print(grad(pick, [10.0, 20.0, 30.0, 40.0]))

acc = 0.0
for i in 0..6:
  acc = acc + sum(slice([10.0, 20.0, 30.0], mod(i, 3.0), 1))
print(acc)
print(mod(-1.0, 3.0))
print(mod([5.0, 6.0, 7.0], 3.0))
```

Output:

```
[20, 30] : f32
[10, 20, 30] : f32
[[5, 6]] : f32
[1, -2, 3] : f32
[0, 1, 1, 0] : f32
120 : f32
2 : f32
[2, 0, 1] : f32
```

## train

```python
fn loss(w):
  ys = load("tests/cases/data/v.npy")
  d = w - ys
  mean(d * d)

w = f64([0.0, 0.0])
for step in 0..10:
  w = w - 1.0 * grad(loss, w)
print(w)
print(loss(w))
```

Output:

```
[3, 4] : f64
0 : f64
```

## transpose

```python
m = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
print(transpose(m))

fn f(x):
  sum(transpose(x) * [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])

print(grad(f, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))

fn t(r):
  transpose(r)

print(vmap(t, [[[1.0, 2.0]], [[3.0, 4.0]]]))
```

Output:

```
[[1, 4], [2, 5], [3, 6]] : f32
[[1, 2, 3], [4, 5, 6]] : f32
[[[1], [2]], [[3], [4]]] : f32
```

## vmap

```python
fn double(x):
  x * 2.0

print(vmap(double, [1.0, 2.0, 3.0]))

fn nsq(v):
  sum(v * v)

print(vmap(nsq, [[3.0, 4.0], [6.0, 8.0]]))

fn dotp(a, b):
  sum(a * b)

print(vmap(dotp, [[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]))

fn lin(x):
  matmul([[1.0, 0.0], [0.0, 2.0]], x)

print(vmap(lin, [[1.0, 1.0], [2.0, 3.0]]))

fn dnsq(v):
  grad(nsq, v)

print(vmap(dnsq, [[3.0, 4.0], [1.0, 2.0]]))

fn agg(m):
  sum(vmap(nsq, m))

print(grad(agg, [[1.0, 2.0], [3.0, 4.0]]))

fn perl(x):
  sum(matmul([[1.0, 2.0], [3.0, 4.0]], x))

fn aggl(m):
  sum(vmap(perl, m))

print(grad(aggl, [[1.0, 1.0], [1.0, 1.0]]))
```

Output:

```
[2, 4, 6] : f32
[25, 100] : f32
[17, 53] : f32
[[1, 2], [2, 6]] : f32
[[6, 8], [2, 4]] : f32
[[2, 4], [6, 8]] : f32
[[4, 6], [4, 6]] : f32
```

## vmap_nested

```python
fn cell(c):
  c * c

fn row(r):
  vmap(cell, r)

print(vmap(row, [[1.0, 2.0], [3.0, 4.0]]))

fn rowsum(r):
  sum(vmap(cell, r))

print(vmap(rowsum, [[1.0, 2.0], [3.0, 4.0]]))

fn aggn(m):
  sum(vmap(rowsum, m))

print(grad(aggn, [[1.0, 2.0], [3.0, 4.0]]))
```

Output:

```
[[1, 4], [9, 16]] : f32
[5, 25] : f32
[[2, 4], [6, 8]] : f32
```
