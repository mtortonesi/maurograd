/*
 * maurograd_mlx: Ruby bindings for the subset of mlx-c (the C API of Apple's
 * MLX array framework) that Maurograd's MLX backend needs.
 *
 * - Maurograd::MLX::Array wraps one mlx_array. Ruby's GC owns the wrapper
 *   and releases the mlx_array when the wrapper is collected. Arithmetic
 *   operators (+ - * / ** and unary -) work between arrays and with Ruby
 *   Numerics on either side, as the backend contract requires.
 * - Module functions on Maurograd::MLX expose the MLX primitives used by
 *   lib/maurograd/backends/mlx.rb, plus conversion to/from Numo arrays,
 *   evaluation, and vjp (vector-Jacobian product) of a Ruby block.
 *
 * MLX is lazy: operations only record a graph, and computation happens
 * when an array is evaluated (MLX.eval, or reading it back with to_numo).
 *
 * Errors: MLX reports failures through an error handler plus a non-zero
 * return code. We install a handler that only records the message; every
 * call checks the code and raises Maurograd::MLX::Error from plain C. We
 * never raise (longjmp) from inside a callback MLX's C++ code is running.
 */
#include "ruby.h"
#include "numo/narray.h"
#include "mlx/c/mlx.h"

#include <stdio.h>
#include <string.h>

static VALUE mMaurograd, mMLX, cArray, eMLXError;
static VALUE cSFloat, cInt32;

static mlx_stream g_stream;     /* stream of the current device */
static int g_gpu;               /* 1 if the current device is the GPU */

/* ---------- error handling ---------- */

static char g_err[2048];
static int g_err_set;

static void on_mlx_error(const char *msg, void *data)
{
    (void)data;
    snprintf(g_err, sizeof(g_err), "%s", msg ? msg : "unknown MLX error");
    g_err_set = 1;
}

NORETURN(static void raise_mlx(const char *what));
static void raise_mlx(const char *what)
{
    if (g_err_set) {
        g_err_set = 0;
        rb_raise(eMLXError, "%s: %s", what, g_err);
    }
    rb_raise(eMLXError, "%s failed", what);
}

/* ---------- Array wrapper ---------- */

typedef struct {
    mlx_array a;
    size_t accounted; /* bytes reported to Ruby's GC */
} mg_array;

static void array_free(void *p)
{
    mg_array *w = (mg_array *)p;
    if (w->a.ctx) mlx_array_free(w->a);
    if (w->accounted) rb_gc_adjust_memory_usage(-(ssize_t)w->accounted);
    xfree(w);
}

static size_t array_memsize(const void *p)
{
    const mg_array *w = (const mg_array *)p;
    return sizeof(*w) + w->accounted;
}

static const rb_data_type_t array_type = {
    "Maurograd::MLX::Array",
    { 0, array_free, array_memsize, },
    0, 0, RUBY_TYPED_FREE_IMMEDIATELY
};

static VALUE array_alloc(VALUE klass)
{
    mg_array *w;
    VALUE obj = TypedData_Make_Struct(klass, mg_array, &array_type, w);
    w->a = mlx_array_new();
    w->accounted = 0;
    return obj;
}

/* Wraps `a`, taking ownership of it. Tells Ruby's GC about the memory the
 * array will use, so that GC runs often enough to release device buffers
 * (Ruby can't see them otherwise and would let them pile up). */
static VALUE wrap(mlx_array a)
{
    VALUE obj = array_alloc(cArray);
    mg_array *w;
    TypedData_Get_Struct(obj, mg_array, &array_type, w);
    w->a = a;
    w->accounted = mlx_array_nbytes(a);
    if (w->accounted) rb_gc_adjust_memory_usage((ssize_t)w->accounted);
    return obj;
}

static int is_array(VALUE v)
{
    return rb_typeddata_is_kind_of(v, &array_type);
}

/* Borrowed mlx_array of a wrapper (raises TypeError otherwise). */
static mlx_array get(VALUE v)
{
    mg_array *w;
    TypedData_Get_Struct(v, mg_array, &array_type, w);
    return w->a;
}

/* Array or Numeric argument. Sets *out; returns 1 if the caller owns (and
 * must free) it, i.e. if it was created from a Numeric. Scalars become
 * float32, so they never promote float32 arrays to float64. */
static int arg_array(VALUE v, mlx_array *out)
{
    if (is_array(v)) {
        *out = get(v);
        return 0;
    }
    if (RB_FLOAT_TYPE_P(v) || RB_INTEGER_TYPE_P(v)) {
        *out = mlx_array_new_float((float)NUM2DBL(v));
        return 1;
    }
    rb_raise(rb_eTypeError, "expected Maurograd::MLX::Array or Numeric, got %s", rb_obj_classname(v));
    return 0;
}

/* Converts a Ruby Array of Integers into a C int array (max 16 entries). */
static size_t int_list(VALUE ary, int *out, size_t cap, const char *what)
{
    Check_Type(ary, T_ARRAY);
    size_t n = (size_t)RARRAY_LEN(ary);
    if (n > cap) rb_raise(rb_eArgError, "too many %s (max %zu)", what, cap);
    for (size_t i = 0; i < n; i++) out[i] = NUM2INT(rb_ary_entry(ary, (long)i));
    return n;
}

typedef int (*binary_fn)(mlx_array *, const mlx_array, const mlx_array, const mlx_stream);
typedef int (*unary_fn)(mlx_array *, const mlx_array, const mlx_stream);

static VALUE binary(VALUE va, VALUE vb, binary_fn f, const char *what)
{
    mlx_array a, b;
    int own_a = arg_array(va, &a);
    int own_b = arg_array(vb, &b);
    mlx_array r = mlx_array_new();
    int rc = f(&r, a, b, g_stream);
    if (own_a) mlx_array_free(a);
    if (own_b) mlx_array_free(b);
    if (rc) { mlx_array_free(r); raise_mlx(what); }
    return wrap(r);
}

static VALUE unary(VALUE va, unary_fn f, const char *what)
{
    mlx_array r = mlx_array_new();
    if (f(&r, get(va), g_stream)) { mlx_array_free(r); raise_mlx(what); }
    return wrap(r);
}

static VALUE astype(mlx_array a, mlx_dtype dt)
{
    mlx_array r = mlx_array_new();
    if (mlx_astype(&r, a, dt, g_stream)) { mlx_array_free(r); raise_mlx("astype"); }
    return wrap(r);
}

/* Comparison results are bool in MLX; the backend wants float 0/1. */
static VALUE compare_as_float(VALUE va, VALUE vb, binary_fn f, const char *what)
{
    VALUE b = binary(va, vb, f, what);
    return astype(get(b), MLX_FLOAT32);
}

/* ---------- Array methods ---------- */

static VALUE arr_add(VALUE self, VALUE o) { return binary(self, o, mlx_add, "add"); }
static VALUE arr_sub(VALUE self, VALUE o) { return binary(self, o, mlx_subtract, "subtract"); }
static VALUE arr_mul(VALUE self, VALUE o) { return binary(self, o, mlx_multiply, "multiply"); }
static VALUE arr_div(VALUE self, VALUE o) { return binary(self, o, mlx_divide, "divide"); }
static VALUE arr_pow(VALUE self, VALUE o) { return binary(self, o, mlx_power, "power"); }
static VALUE arr_neg(VALUE self) { return unary(self, mlx_negative, "negative"); }

/* `2.0 * arr`: Float#* calls arr.coerce(2.0) and retries as scalar * arr. */
static VALUE arr_coerce(VALUE self, VALUE other)
{
    mlx_array s;
    arg_array(other, &s); /* Numeric -> new float32 scalar array */
    return rb_assoc_new(wrap(s), self);
}

static VALUE arr_shape(VALUE self)
{
    mlx_array a = get(self);
    size_t nd = mlx_array_ndim(a);
    const int *sh = mlx_array_shape(a);
    VALUE ary = rb_ary_new_capa((long)nd);
    for (size_t i = 0; i < nd; i++) rb_ary_push(ary, INT2NUM(sh[i]));
    return ary;
}

static VALUE arr_size(VALUE self) { return SIZET2NUM(mlx_array_size(get(self))); }

static VALUE arr_dtype(VALUE self)
{
    switch (mlx_array_dtype(get(self))) {
    case MLX_FLOAT32: return ID2SYM(rb_intern("float32"));
    case MLX_INT32: return ID2SYM(rb_intern("int32"));
    case MLX_BOOL: return ID2SYM(rb_intern("bool"));
    default: return ID2SYM(rb_intern("other"));
    }
}

/* ---------- conversion ---------- */

static VALUE mlx_from_numo(VALUE self, VALUE vna)
{
    mlx_dtype dt;
    if (rb_obj_is_kind_of(vna, cSFloat)) dt = MLX_FLOAT32;
    else if (rb_obj_is_kind_of(vna, cInt32)) dt = MLX_INT32;
    else rb_raise(rb_eTypeError, "from_numo needs a Numo::SFloat or Numo::Int32");
    if (!RTEST(na_check_contiguous(vna))) rb_raise(rb_eArgError, "from_numo needs a contiguous array");

    narray_t *na;
    GetNArray(vna, na);
    int shape[16];
    if (na->ndim > 16) rb_raise(rb_eArgError, "too many dimensions");
    for (int i = 0; i < na->ndim; i++) shape[i] = (int)na->shape[i];

    const void *ptr = na_get_pointer_for_read(vna);
    /* mlx_array_new_data copies the buffer, so the Numo array can go away. */
    mlx_array a = mlx_array_new_data(ptr, shape, na->ndim, dt);
    RB_GC_GUARD(vna);
    return wrap(a);
}

/* Evaluates `va` and copies it into a new Numo array (SFloat, or Int32 for
 * int32 arrays). Other dtypes are converted to float32 first. */
static VALUE mlx_to_numo(VALUE self, VALUE va)
{
    mlx_array a = get(va);
    mlx_dtype dt = mlx_array_dtype(a);
    VALUE keep = Qnil;
    if (dt != MLX_FLOAT32 && dt != MLX_INT32) {
        keep = astype(a, MLX_FLOAT32);
        a = get(keep);
        dt = MLX_FLOAT32;
    }

    /* Results can be strided views (e.g. after a transpose): make the
     * buffer row-major before reading it. */
    mlx_array c = mlx_array_new();
    if (mlx_contiguous(&c, a, false, g_stream)) { mlx_array_free(c); raise_mlx("contiguous"); }
    VALUE vc = wrap(c);
    if (mlx_array_eval(c)) raise_mlx("eval");

    size_t nd = mlx_array_ndim(c);
    const int *sh = mlx_array_shape(c);
    size_t shape[16];
    if (nd > 16) rb_raise(rb_eArgError, "too many dimensions");
    for (size_t i = 0; i < nd; i++) shape[i] = (size_t)sh[i];

    VALUE out = nary_new(dt == MLX_FLOAT32 ? cSFloat : cInt32, (int)nd, shape);
    size_t n = mlx_array_size(c);
    void *dst = na_get_pointer_for_write(out);
    const void *src = (dt == MLX_FLOAT32) ? (const void *)mlx_array_data_float32(c)
                                          : (const void *)mlx_array_data_int32(c);
    if (n && !src) rb_raise(eMLXError, "could not read array data");
    if (n) memcpy(dst, src, n * 4);

    RB_GC_GUARD(vc);
    RB_GC_GUARD(keep);
    RB_GC_GUARD(va);
    return out;
}

static VALUE arr_to_numo(VALUE self) { return mlx_to_numo(Qnil, self); }

/* ---------- creation, randomness ---------- */

static VALUE filled(VALUE vshape, int ones)
{
    int shape[16];
    size_t n = int_list(vshape, shape, 16, "dimensions");
    mlx_array r = mlx_array_new();
    int rc = ones ? mlx_ones(&r, shape, n, MLX_FLOAT32, g_stream)
                  : mlx_zeros(&r, shape, n, MLX_FLOAT32, g_stream);
    if (rc) { mlx_array_free(r); raise_mlx(ones ? "ones" : "zeros"); }
    return wrap(r);
}

static VALUE mlx_zeros_rb(VALUE self, VALUE shape) { return filled(shape, 0); }
static VALUE mlx_ones_rb(VALUE self, VALUE shape) { return filled(shape, 1); }

static VALUE mlx_bernoulli_rb(VALUE self, VALUE vshape, VALUE vp)
{
    int shape[16];
    size_t n = int_list(vshape, shape, 16, "dimensions");
    mlx_array p = mlx_array_new_float((float)NUM2DBL(vp));
    mlx_array key = { 0 }; /* null: use the global key (see seed) */
    mlx_array r = mlx_array_new();
    int rc = mlx_random_bernoulli(&r, p, shape, n, key, g_stream);
    mlx_array_free(p);
    if (rc) { mlx_array_free(r); raise_mlx("bernoulli"); }
    VALUE b = wrap(r);
    return astype(get(b), MLX_FLOAT32);
}

static VALUE mlx_seed_rb(VALUE self, VALUE n)
{
    if (mlx_random_seed(NUM2ULL(n))) raise_mlx("seed");
    return n;
}

/* ---------- shape ---------- */

static VALUE mlx_reshape_rb(VALUE self, VALUE va, VALUE vshape)
{
    int shape[16];
    size_t n = int_list(vshape, shape, 16, "dimensions");
    mlx_array r = mlx_array_new();
    if (mlx_reshape(&r, get(va), shape, n, g_stream)) { mlx_array_free(r); raise_mlx("reshape"); }
    return wrap(r);
}

static VALUE mlx_transpose_rb(VALUE self, VALUE va, VALUE vaxes)
{
    int axes[16];
    size_t n = int_list(vaxes, axes, 16, "axes");
    mlx_array r = mlx_array_new();
    if (mlx_transpose_axes(&r, get(va), axes, n, g_stream)) { mlx_array_free(r); raise_mlx("transpose"); }
    return wrap(r);
}

/* x[start:stop:strides] along every axis. */
static VALUE mlx_slice_rb(VALUE self, VALUE va, VALUE vstart, VALUE vstop, VALUE vstrides)
{
    int start[16], stop[16], strides[16];
    size_t n0 = int_list(vstart, start, 16, "start");
    size_t n1 = int_list(vstop, stop, 16, "stop");
    size_t n2 = int_list(vstrides, strides, 16, "strides");
    mlx_array r = mlx_array_new();
    if (mlx_slice(&r, get(va), start, n0, stop, n1, strides, n2, g_stream)) { mlx_array_free(r); raise_mlx("slice"); }
    return wrap(r);
}

/* Stacks a Ruby Array of arrays along a new leading axis. */
static VALUE mlx_stack_rb(VALUE self, VALUE vlist)
{
    Check_Type(vlist, T_ARRAY);
    mlx_vector_array v = mlx_vector_array_new();
    for (long i = 0; i < RARRAY_LEN(vlist); i++) mlx_vector_array_append_value(v, get(rb_ary_entry(vlist, i)));
    mlx_array r = mlx_array_new();
    int rc = mlx_stack(&r, v, g_stream);
    mlx_vector_array_free(v);
    if (rc) { mlx_array_free(r); raise_mlx("stack"); }
    return wrap(r);
}

/* Constant padding of the given axes. */
static VALUE mlx_pad_rb(VALUE self, VALUE va, VALUE vaxes, VALUE vlow, VALUE vhigh, VALUE vvalue)
{
    int axes[16], low[16], high[16];
    size_t n0 = int_list(vaxes, axes, 16, "axes");
    size_t n1 = int_list(vlow, low, 16, "low");
    size_t n2 = int_list(vhigh, high, 16, "high");
    mlx_array value = mlx_array_new_float((float)NUM2DBL(vvalue));
    mlx_array r = mlx_array_new();
    int rc = mlx_pad(&r, get(va), axes, n0, low, n1, high, n2, value, "constant", g_stream);
    mlx_array_free(value);
    if (rc) { mlx_array_free(r); raise_mlx("pad"); }
    return wrap(r);
}

/* ---------- reductions ---------- */

typedef int (*reduce_all_fn)(mlx_array *, const mlx_array, bool, const mlx_stream);
typedef int (*reduce_axes_fn)(mlx_array *, const mlx_array, const int *, size_t, bool, const mlx_stream);

static VALUE reduce(VALUE va, VALUE vaxes, VALUE vkeep, reduce_all_fn all, reduce_axes_fn some, const char *what)
{
    mlx_array r = mlx_array_new();
    int rc;
    if (NIL_P(vaxes)) {
        rc = all(&r, get(va), RTEST(vkeep), g_stream);
    } else {
        int axes[16];
        size_t n = int_list(rb_Array(vaxes), axes, 16, "axes");
        rc = some(&r, get(va), axes, n, RTEST(vkeep), g_stream);
    }
    if (rc) { mlx_array_free(r); raise_mlx(what); }
    return wrap(r);
}

static VALUE mlx_sum_rb(VALUE self, VALUE a, VALUE axes, VALUE keep) { return reduce(a, axes, keep, mlx_sum, mlx_sum_axes, "sum"); }
static VALUE mlx_mean_rb(VALUE self, VALUE a, VALUE axes, VALUE keep) { return reduce(a, axes, keep, mlx_mean, mlx_mean_axes, "mean"); }
static VALUE mlx_max_rb(VALUE self, VALUE a, VALUE axes, VALUE keep) { return reduce(a, axes, keep, mlx_max, mlx_max_axes, "max"); }

/* ---------- elementwise / linear algebra ---------- */

static VALUE mlx_exp_rb(VALUE self, VALUE a) { return unary(a, mlx_exp, "exp"); }
static VALUE mlx_log_rb(VALUE self, VALUE a) { return unary(a, mlx_log, "log"); }
static VALUE mlx_sqrt_rb(VALUE self, VALUE a) { return unary(a, mlx_sqrt, "sqrt"); }
static VALUE mlx_maximum_rb(VALUE self, VALUE a, VALUE b) { return binary(a, b, mlx_maximum, "maximum"); }
static VALUE mlx_greater_rb(VALUE self, VALUE a, VALUE b) { return compare_as_float(a, b, mlx_greater, "greater"); }
static VALUE mlx_equal_rb(VALUE self, VALUE a, VALUE b) { return compare_as_float(a, b, mlx_equal, "equal"); }
static VALUE mlx_matmul_rb(VALUE self, VALUE a, VALUE b) { return binary(a, b, mlx_matmul, "matmul"); }

/* Index of the first maximum along `axis` (int32). */
static VALUE mlx_argmax_rb(VALUE self, VALUE va, VALUE vaxis)
{
    mlx_array r = mlx_array_new();
    if (mlx_argmax_axis(&r, get(va), NUM2INT(vaxis), false, g_stream)) { mlx_array_free(r); raise_mlx("argmax"); }
    VALUE idx = wrap(r);
    return astype(get(idx), MLX_INT32);
}

/* [0, 1, ..., n-1] as int32. */
static VALUE mlx_arange_rb(VALUE self, VALUE vn)
{
    mlx_array r = mlx_array_new();
    if (mlx_arange(&r, 0, NUM2INT(vn), 1, MLX_INT32, g_stream)) { mlx_array_free(r); raise_mlx("arange"); }
    return wrap(r);
}

/* Float [N, C] one-hot rows: (indices[:, None] == arange(C)). */
static VALUE mlx_one_hot_rb(VALUE self, VALUE vidx, VALUE vc)
{
    VALUE idx = astype(get(vidx), MLX_INT32);
    mlx_array col = mlx_array_new();
    if (mlx_expand_dims(&col, get(idx), 1, g_stream)) { mlx_array_free(col); raise_mlx("expand_dims"); }
    VALUE vcol = wrap(col);
    mlx_array range = mlx_array_new();
    if (mlx_arange(&range, 0, NUM2INT(vc), 1, MLX_INT32, g_stream)) { mlx_array_free(range); raise_mlx("arange"); }
    VALUE vrange = wrap(range);
    VALUE out = compare_as_float(vcol, vrange, mlx_equal, "equal");
    RB_GC_GUARD(idx);
    return out;
}

/* Channels-last convolution: x [N,H,W,C], w [Cout,KH,KW,C] -> [N,OH,OW,Cout]. */
static VALUE mlx_conv2d_rb(VALUE self, VALUE vx, VALUE vw, VALUE vstride, VALUE vpad)
{
    int s = NUM2INT(vstride), p = NUM2INT(vpad);
    mlx_array r = mlx_array_new();
    if (mlx_conv2d(&r, get(vx), get(vw), s, s, p, p, 1, 1, 1, g_stream)) { mlx_array_free(r); raise_mlx("conv2d"); }
    return wrap(r);
}

/* ---------- evaluation ---------- */

static mlx_vector_array vector_of(VALUE list)
{
    Check_Type(list, T_ARRAY);
    mlx_vector_array v = mlx_vector_array_new();
    for (long i = 0; i < RARRAY_LEN(list); i++) {
        VALUE e = rb_ary_entry(list, i);
        if (!NIL_P(e)) mlx_vector_array_append_value(v, get(e));
    }
    return v;
}

/* Computes every array in the list now. */
static VALUE mlx_eval_rb(VALUE self, VALUE list)
{
    mlx_vector_array v = vector_of(list);
    int rc = mlx_eval(v);
    mlx_vector_array_free(v);
    if (rc) raise_mlx("eval");
    return Qnil;
}

/* Schedules the computation of every array in the list, without waiting. */
static VALUE mlx_async_eval_rb(VALUE self, VALUE list)
{
    mlx_vector_array v = vector_of(list);
    int rc = mlx_async_eval(v);
    mlx_vector_array_free(v);
    if (rc) raise_mlx("async_eval");
    return Qnil;
}

/* ---------- vjp of a Ruby block ---------- */

/*
 * MLX.vjp(primals, cotangents) { |*xs| outputs } -> gradients
 *
 * Runs MLX's reverse-mode differentiation of the function the block
 * defines: the block receives one MLX::Array per primal and returns an
 * Array of outputs; the result holds d(sum(outputs * cotangents))/d(primal)
 * for each primal. The block runs synchronously, on this thread, while MLX
 * traces it inside mlx_vjp, so calling Ruby is safe there; exceptions are
 * caught with rb_protect and re-raised after MLX has returned.
 */
typedef struct {
    VALUE block;
    int state;  /* Ruby exception state from the block, if any */
    int bad;    /* block returned something that isn't Array of MLX::Array */
} vjp_payload;

static VALUE call_block(VALUE args)
{
    return rb_proc_call(rb_ary_entry(args, 0), rb_ary_entry(args, 1));
}

static int vjp_closure(mlx_vector_array *res, const mlx_vector_array in, void *payload)
{
    vjp_payload *p = (vjp_payload *)payload;
    size_t n = mlx_vector_array_size(in);
    VALUE inputs = rb_ary_new_capa((long)n);
    for (size_t i = 0; i < n; i++) {
        mlx_array a = mlx_array_new();
        mlx_vector_array_get(&a, in, i);
        rb_ary_push(inputs, wrap(a));
    }

    VALUE args = rb_assoc_new(p->block, inputs);
    int state = 0;
    VALUE out = rb_protect(call_block, args, &state);
    if (state) { p->state = state; return 1; }

    if (!RB_TYPE_P(out, T_ARRAY)) { p->bad = 1; return 1; }
    mlx_vector_array v = mlx_vector_array_new();
    for (long i = 0; i < RARRAY_LEN(out); i++) {
        VALUE e = rb_ary_entry(out, i);
        if (!is_array(e)) { mlx_vector_array_free(v); p->bad = 1; return 1; }
        mlx_vector_array_append_value(v, get(e));
    }
    mlx_vector_array_set(res, v);
    mlx_vector_array_free(v);
    RB_GC_GUARD(inputs);
    RB_GC_GUARD(out);
    return 0;
}

static VALUE mlx_vjp_rb(VALUE self, VALUE vprimals, VALUE vcotangents)
{
    if (!rb_block_given_p()) rb_raise(rb_eArgError, "vjp needs a block");
    vjp_payload payload = { rb_block_proc(), 0, 0 };

    mlx_closure fn = mlx_closure_new_func_payload(vjp_closure, &payload, NULL);
    mlx_vector_array primals = vector_of(vprimals);
    mlx_vector_array cotangents = vector_of(vcotangents);
    mlx_vector_array outs = mlx_vector_array_new();
    mlx_vector_array grads = mlx_vector_array_new();

    int rc = mlx_vjp(&outs, &grads, fn, primals, cotangents);

    mlx_closure_free(fn);
    mlx_vector_array_free(primals);
    mlx_vector_array_free(cotangents);
    mlx_vector_array_free(outs);

    if (payload.state) { mlx_vector_array_free(grads); g_err_set = 0; rb_jump_tag(payload.state); }
    if (payload.bad) { mlx_vector_array_free(grads); g_err_set = 0; rb_raise(rb_eTypeError, "vjp block must return an Array of Maurograd::MLX::Array"); }
    if (rc) { mlx_vector_array_free(grads); raise_mlx("vjp"); }

    size_t n = mlx_vector_array_size(grads);
    VALUE result = rb_ary_new_capa((long)n);
    for (size_t i = 0; i < n; i++) {
        mlx_array g = mlx_array_new();
        mlx_vector_array_get(&g, grads, i);
        rb_ary_push(result, wrap(g));
    }
    mlx_vector_array_free(grads);
    RB_GC_GUARD(payload.block);
    return result;
}

/* ---------- device ---------- */

static void set_device(int gpu)
{
    mlx_device dev = mlx_device_new_type(gpu ? MLX_GPU : MLX_CPU, 0);
    if (mlx_set_default_device(dev)) { mlx_device_free(dev); raise_mlx("set_default_device"); }
    mlx_stream s = mlx_stream_new();
    if (mlx_get_default_stream(&s, dev)) { mlx_stream_free(s); mlx_device_free(dev); raise_mlx("get_default_stream"); }
    mlx_device_free(dev);
    if (g_stream.ctx) mlx_stream_free(g_stream);
    g_stream = s;
    g_gpu = gpu;
}

/* MLX.device -> :gpu or :cpu */
static VALUE mlx_device_rb(VALUE self) { return ID2SYM(rb_intern(g_gpu ? "gpu" : "cpu")); }

/* MLX.device = :gpu / :cpu. Arrays already created keep working. */
static VALUE mlx_set_device_rb(VALUE self, VALUE vdev)
{
    ID id = SYM2ID(rb_to_symbol(vdev));
    if (id == rb_intern("gpu")) {
        bool ok = false;
        mlx_metal_is_available(&ok);
        if (!ok) rb_raise(eMLXError, "no Metal GPU available");
        set_device(1);
    } else if (id == rb_intern("cpu")) {
        set_device(0);
    } else {
        rb_raise(rb_eArgError, "device must be :gpu or :cpu");
    }
    return vdev;
}

void Init_maurograd_mlx(void)
{
    VALUE cNumo = rb_const_get(rb_cObject, rb_intern("Numo"));
    cSFloat = rb_const_get(cNumo, rb_intern("SFloat"));
    cInt32 = rb_const_get(cNumo, rb_intern("Int32"));

    mMaurograd = rb_define_module("Maurograd");
    mMLX = rb_define_module_under(mMaurograd, "MLX");
    eMLXError = rb_define_class_under(mMLX, "Error", rb_eStandardError);

    mlx_set_error_handler(on_mlx_error, NULL, NULL);

    g_stream.ctx = NULL;
    bool metal = false;
    mlx_metal_is_available(&metal);
    set_device(metal ? 1 : 0);

    cArray = rb_define_class_under(mMLX, "Array", rb_cObject);
    rb_undef_alloc_func(cArray);
    rb_define_method(cArray, "+", arr_add, 1);
    rb_define_method(cArray, "-", arr_sub, 1);
    rb_define_method(cArray, "*", arr_mul, 1);
    rb_define_method(cArray, "/", arr_div, 1);
    rb_define_method(cArray, "**", arr_pow, 1);
    rb_define_method(cArray, "-@", arr_neg, 0);
    rb_define_method(cArray, "coerce", arr_coerce, 1);
    rb_define_method(cArray, "shape", arr_shape, 0);
    rb_define_method(cArray, "size", arr_size, 0);
    rb_define_method(cArray, "dtype", arr_dtype, 0);
    rb_define_method(cArray, "to_numo", arr_to_numo, 0);

    rb_define_module_function(mMLX, "from_numo", mlx_from_numo, 1);
    rb_define_module_function(mMLX, "to_numo", mlx_to_numo, 1);
    rb_define_module_function(mMLX, "zeros", mlx_zeros_rb, 1);
    rb_define_module_function(mMLX, "ones", mlx_ones_rb, 1);
    rb_define_module_function(mMLX, "bernoulli", mlx_bernoulli_rb, 2);
    rb_define_module_function(mMLX, "seed", mlx_seed_rb, 1);
    rb_define_module_function(mMLX, "reshape", mlx_reshape_rb, 2);
    rb_define_module_function(mMLX, "transpose", mlx_transpose_rb, 2);
    rb_define_module_function(mMLX, "slice", mlx_slice_rb, 4);
    rb_define_module_function(mMLX, "stack", mlx_stack_rb, 1);
    rb_define_module_function(mMLX, "pad", mlx_pad_rb, 5);
    rb_define_module_function(mMLX, "sum", mlx_sum_rb, 3);
    rb_define_module_function(mMLX, "mean", mlx_mean_rb, 3);
    rb_define_module_function(mMLX, "max", mlx_max_rb, 3);
    rb_define_module_function(mMLX, "exp", mlx_exp_rb, 1);
    rb_define_module_function(mMLX, "log", mlx_log_rb, 1);
    rb_define_module_function(mMLX, "sqrt", mlx_sqrt_rb, 1);
    rb_define_module_function(mMLX, "maximum", mlx_maximum_rb, 2);
    rb_define_module_function(mMLX, "greater", mlx_greater_rb, 2);
    rb_define_module_function(mMLX, "equal", mlx_equal_rb, 2);
    rb_define_module_function(mMLX, "argmax", mlx_argmax_rb, 2);
    rb_define_module_function(mMLX, "arange", mlx_arange_rb, 1);
    rb_define_module_function(mMLX, "matmul", mlx_matmul_rb, 2);
    rb_define_module_function(mMLX, "one_hot", mlx_one_hot_rb, 2);
    rb_define_module_function(mMLX, "conv2d", mlx_conv2d_rb, 4);
    rb_define_module_function(mMLX, "eval", mlx_eval_rb, 1);
    rb_define_module_function(mMLX, "async_eval", mlx_async_eval_rb, 1);
    rb_define_module_function(mMLX, "vjp", mlx_vjp_rb, 2);
    rb_define_module_function(mMLX, "device", mlx_device_rb, 0);
    rb_define_module_function(mMLX, "device=", mlx_set_device_rb, 1);
}
