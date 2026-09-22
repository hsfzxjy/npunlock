# MoviTools math-library symbol inventory

Status: observed inventory. Last updated 2026-09-22.

This page lists the externally defined symbols in the `mlibm.a` shipped with
the MoviTools package documented by this project. The inspected archive has
SHA-256:

```text
4769ac96201c3fbd4c974151a631a42b8339b30ea26d8f54072375092bda54a3
```

These symbol names are candidates that a custom kernel may reference. A name
in the archive does not by itself prove that the function works within
`npunlock`'s kernel contract. MoviTools must accept the call, the linker must
resolve it, section garbage collection must leave a self-contained image, and
the resulting ELF must pass validation.

The archive symbol table does not describe C parameter or return types. The
signatures below are therefore not determined. For familiar names, a likely
prototype can be inferred from the conventional `libm` function name or from
an existing `libm` implementation, but that remains an inference until the
kernel compiles and its output is tested against a host oracle. Symbols with
implementation-style names, especially those beginning with `__`, need extra
caution.

The inventory was produced with LLVM 20.1.7:

```text
llvm-nm --defined-only --extern-only --format=posix mlibm.a
```

It contains 334 symbols: 309 `T` (code) symbols and 25 `D` (initialized data)
symbols. `T` entries are potential callable functions. `D` entries are not
functions; retaining their storage may conflict with the current requirement
that the linked kernel's `.arg.data` section remain empty. Sizes are hexadecimal
and values are zero because these are relocatable archive members.

## Symbol dump

```text
halfmath.o:
__acoshs T 0 34a
__acoss T 0 12b
__asinhs T 0 1b2
__asins T 0 129
__atan2s T 0 263
__atanhs T 0 17c
__atans T 0 102
__cbrts T 0 82
__ceils T 0 2c
__copysigns T 0 17
__coshs T 0 12a
__coss T 0 1a5
__erfcs T 0 581
__erfs T 0 1fc
__exp2s T 0 b9
__expm1s T 0 22c
__exps T 0 1c5
__fdims T 0 46
__float_to__fp16_denormal_support T 0 b6
__floors T 0 46
__fmas T 0 1e
__fmaxs T 0 31
__fmins T 0 31
__fmods T 0 1a1
__fp16_to_float_denormal_support T 0 62
__frexps T 0 57
__hypots T 0 11b
__ilogbs T 0 39
__ldexps T 0 2e
__lgammas T 0 ead
__log10s T 0 13c
__log1ps T 0 216
__logbs T 0 3a
__logs T 0 13c
__modfs T 0 44
__nearbyints T 0 1f3
__nextafters T 0 93
__nexttowards T 0 e7
__powrs T 0 1a7
__pows T 0 43e
__remainders T 0 232
__remquos T 0 28f
__rounds T 0 3b
__scalblns T 0 2e
__scalbns T 0 2e
__sinhs T 0 152
__sins T 0 194
__tanhs T 0 131
__tans T 0 16a
__tgammas T 0 834
__truncs T 0 2c

math_lite.o:
__comparelt T 0 18
__comparelte T 0 18
__powr T 0 35e
__powrf T 0 35e
__remquo_pos_only T 0 245
__tgamma_positive_4_upf T 0 129
__tgamma_positivef T 0 217
acos T 0 181
acosf T 0 181
acosh T 0 14e
acoshf T 0 14e
asin T 0 1b2
asinf T 0 1b2
asinh T 0 52e
asinhf T 0 52e
atan T 0 131
atan2 T 0 208
atan2f T 0 208
atanf T 0 131
atanh T 0 110
atanhf T 0 110
cbrt T 0 e1
cbrtf T 0 e1
ceil T 0 3b
ceilf T 0 3b
copysign T 0 c
copysignf T 0 c
cos T 0 18d
cosf T 0 18d
cosh T 0 85
coshf T 0 85
erf T 0 5ed
erfc T 0 94d
erfcf T 0 94d
erff T 0 5ed
exp T 0 136
exp2 T 0 b8
exp2f T 0 b8
expf T 0 136
expm1 T 0 c0
expm1f T 0 c0
fabs T 0 c
fabsf T 0 c
fdim T 0 57
fdimf T 0 57
floor T 0 44
floorf T 0 44
fma T 0 615
fmaf T 0 615
fmax T 0 30
fmaxf T 0 30
fmin T 0 30
fminf T 0 30
fmod T 0 17e
fmodf T 0 17e
frexp T 0 6c
frexpf T 0 6c
hypot T 0 138
hypotf T 0 138
ilogb T 0 37
ilogbf T 0 37
ldexp T 0 af
ldexpf T 0 af
lgamma T 0 912
lgammaf T 0 912
llrint T 0 2d
llrintf T 0 2d
llround T 0 2d
llroundf T 0 2d
log T 0 127
log10 T 0 2e
log10f T 0 2e
log1p T 0 2d4
log1pf T 0 2d4
log2 T 0 1e6
log2f T 0 1e6
logb T 0 40
logbf T 0 40
logf T 0 127
lrint T 0 26
lrintf T 0 26
lround T 0 26
lroundf T 0 26
modf T 0 32
modff T 0 32
nan T 0 2d
nanf T 0 2d
nearbyint T 0 160
nearbyintf T 0 160
nextafter T 0 94
nextafterf T 0 94
nexttoward T 0 e3
nexttowardf T 0 e3
pow T 0 5e2
powf T 0 5e2
remainder T 0 213
remainderf T 0 213
remquo T 0 274
remquof T 0 274
rint T 0 160
rintf T 0 160
round T 0 46
roundf T 0 46
scalbln T 0 af
scalblnf T 0 af
scalbn T 0 af
scalbnf T 0 af
sin T 0 185
sinf T 0 185
sinh T 0 26f
sinhf T 0 26f
sqrt T 0 14c
sqrtf T 0 14c
tan T 0 168
tanf T 0 168
tanh T 0 c9
tanhf T 0 c9
tgamma T 0 37a
tgammaf T 0 37a
trunc T 0 29
truncf T 0 29

shave_fenv.o:
__fe_dfl_env D 0 4
__getfptointround T 0 1e
__setfptointround T 0 42
feclearexcept T 0 1a
fegetenv T 0 1f
fegetexceptflag T 0 41
fegetround T 0 1e
feholdexcept T 0 b
feraiseexcept T 0 1a
fesetenv T 0 34
fesetexceptflag T 0 1a
fesetround T 0 42
fetestexcept T 0 8
feupdateenv T 0 34
roundMode D 0 4

shave_fp64.o:
__add_unsafel T 0 4ab
__clampl T 0 3f
__cospi_reduced T 0 209
__double_to_float_unsafel T 0 26
__exact_sqrtl T 0 6c2
__expm1_unsafe62 T 0 4b3
__float_to_double_unsafe T 0 30
__floor_unsafel T 0 167
__fmaxl_logic_lookup_table T 0 78
__fminl_logic_lookup_table T 0 78
__fpclassifyl T 0 109
__frexpf_unsafefl T 0 39
__log1p62 T 0 7c5
__log62 T 0 57d
__log_extended_precision T 0 6b9
__minus_xl T 0 2e
__modf_unsafel T 0 28d
__mul_unsafe_with_remainderl T 0 628
__mul_unsafel T 0 399
__multiply_mantissas_hi_lo_unsafel T 0 192
__normalise_coefficients T 0 1a8
__normalise_denorml T 0 a4
__poly_evaluation_62_bit_accuracy62 T 0 5b6
__poly_evaluation_62_bit_accuracy_preset_coeffsl T 0 2f0
__poly_evaluation_62_bit_accuracyl T 0 878
__poly_evaluationl T 0 90
__pow_xpositive62 T 0 cf8
__round_unsafel T 0 172
__signbitl T 0 8
__sinpi62 T 0 517
__sinpi_reduced T 0 3fb
__sqrt_unsafe62 T 0 d14
__sqrt_unsafel T 0 f9f
__tgamma_positive_4_upl T 0 1525
__trunc_unsafel T 0 b3
acoshl T 0 86b
acosl T 0 bdb
asinhl T 0 ce4
asinl T 0 9a9
atan2l T 0 a2c
atanhl T 0 770
atanl T 0 61e
cbrtl T 0 c36
ceill T 0 4d
copysignl T 0 2f
coshl T 0 40f
cosl T 0 626
cospi_coeffs D 0 70
erf_one_half_to_two_coeffs D 0 e0
erf_one_to_one_half_coeffs D 0 e0
erf_three_to_three_half_coeffs D 0 b0
erf_two_half_to_three_coeffs D 0 e0
erf_two_to_two_half_coeffs D 0 e0
erf_zero_to_one_coeffs D 0 c0
erfc_coeffs_1062 D 0 f0
erfc_coeffs_1162 D 0 f0
erfc_coeffs_1262 D 0 f0
erfc_coeffs_1362 D 0 f0
erfc_coeffs_162 D 0 b0
erfc_coeffs_262 D 0 d0
erfc_coeffs_362 D 0 d0
erfc_coeffs_462 D 0 d0
erfc_coeffs_562 D 0 e0
erfc_coeffs_662 D 0 e0
erfc_coeffs_762 D 0 e0
erfc_coeffs_862 D 0 f0
erfc_coeffs_962 D 0 f0
erfcl T 0 155d
erfl T 0 1711
exp2l T 0 4ce
exp_unsafe62 T 0 ba0
expl T 0 51c
expm1l T 0 45e
fabsl T 0 2c
fdiml T 0 112
floorl T 0 1c2
fmal T 0 596
fmaxl T 0 ff
fminl T 0 11c
fmodl T 0 393
frexpl T 0 17b
hypotl T 0 25a
ilogbl T 0 1b2
is_denormal_or_zero T 0 36
is_nan_or_infl T 0 1f
isinfl T 0 40
isnanl T 0 49
iszerol T 0 3c
lanczos_ln_gamma62 T 0 e43
ldexpl T 0 429
lgammal T 0 245
llrintl T 0 2d
llroundl T 0 2d
log10l T 0 8d2
log1pl T 0 4d1
log2l T 0 71e
logbl T 0 178
logl T 0 3cc
lrintl T 0 2d
lroundl T 0 2d
modfl T 0 372
nanl T 0 f
nearbyintl T 0 295
nextafterl T 0 260
nexttowardl T 0 260
powl T 0 dbb
range_reduction_payne_hanek T 0 a71
remainderl T 0 26
remquol T 0 73b
rintl T 0 295
roundl T 0 201
scalblnl T 0 22
scalbnl T 0 22
simpl62 D 0 e0
sinhl T 0 a35
sinl T 0 591
sinpi_coeffs D 0 70
sinpi_coeffs62 D 0 80
sqrtl T 0 231
tanhl T 0 e9d
tanl T 0 b9a
tgammal T 0 53b
truncl T 0 11f
x_mod_2_62 T 0 ff
x_times_half_unsafel T 0 2f

shave_fpclassify.o:
__fpclassifyd T 0 63
__fpclassifyf T 0 63
__fpclassifys T 0 5c

shave_signbit.o:

ef_j0.o:
__ieee754_j0f T 0 779
__ieee754_y0f T 0 75d

ef_j1.o:
__ieee754_j1f T 0 543
__ieee754_y1f T 0 5b1

ef_jn.o:
__ieee754_jnf T 0 448
__ieee754_ynf T 0 1e5

wf_j0.o:
j0 T 0 a4
j0f T 0 a4
y0 T 0 e4
y0f T 0 e4

wf_j1.o:
j1 T 0 a8
j1f T 0 a8
y1 T 0 e4
y1f T 0 e4

wf_jn.o:
jn T 0 a4
jnf T 0 a4
yn T 0 e4
ynf T 0 e4

ef_sqrt.o:
__ieee754_sqrtf T 0 860
```

## Related documentation

- [MoviTools DLL invocation contract](MOVITOOLS.md)
- [ACT kernel ELF ABI](KERNEL_ELF.md)
- [Writing custom kernels](../wiki/CUSTOM_KERNELS.md)

[Back to documentation index](../wiki/README.md)
