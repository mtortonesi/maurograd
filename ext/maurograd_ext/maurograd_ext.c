#include "ruby.h"
#include "ruby/thread.h"
#include "numo/narray.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <limits.h>

#include <cblas.h>   // OpenBLAS (cblas_*)

extern void openblas_set_num_threads(int);

/*
 * Minimal helpers: type checks and pointer access.
 * NB: usare le API Numo per ottenere puntatori PRIMA di rilasciare il GVL.
 */

static VALUE mMaurograd;
static VALUE mExt;
static VALUE cNumo;
static VALUE cNArray_;
static VALUE cSFloat_;
static ID id_zeros;
static ID id_col;
static VALUE sym_col;


/* ---------- Helper functions ---------- */
#include <math.h>

static inline float absmax(const float *p, size_t n) {
    float m = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float v = fabsf(p[i]);
        if (v > m) m = v;
    }
    return m;
}


static inline int any_nonfinite(const float *p, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (!isfinite(p[i])) return 1;
    }
    return 0;
}

static inline VALUE sfloat_new_2d(size_t row, size_t col)
{
    size_t shape[2] = { row, col };
    return nary_new(cSFloat_, 2, shape);
}

static inline VALUE sfloat_new_4d(size_t a, size_t b, size_t c, size_t d) {
    size_t shape[4] = { a, b, c, d };
    return nary_new(cSFloat_, 4, shape);
}

static inline int sz_to_int(size_t v, const char *name)
{
    if (v > (size_t)INT_MAX) {
        rb_raise(rb_eArgError, "%s too large: %zu > INT_MAX", name, v);
    }
    return (int)v;
}

/* ---------- reusable GEMM (row-major) ---------- */

static inline void sgemm_rowmajor_nn(int M, int N, int K,
                                    const float *A, const float *B, float *C)
{
    // C = A*B  where:
    // A: (M,K) row-major
    // B: (K,N) row-major
    // C: (M,N) row-major
    const float alpha = 1.0f;
    const float beta  = 0.0f;

    cblas_sgemm(CblasRowMajor,
                CblasNoTrans, CblasNoTrans,
                M, N, K,
                alpha,
                A, K,      // lda = K
                B, N,      // ldb = N
                beta,
                C, N);     // ldc = N
}

static inline void sgemm_rowmajor_nt(int M, int N, int K,
                                    const float *A, const float *B, float *C,
                                    int B_rows, int B_cols)
{
    // C = A * B^T, where:
    // A: (M,K) row-major
    // B: (N,K) row-major  (so B^T is (K,N))
    // C: (M,N) row-major
    //
    // Here B is stored as (N,K) row-major contiguous (B_rows=N, B_cols=K).
    const float alpha = 1.0f;
    const float beta  = 0.0f;

    cblas_sgemm(CblasRowMajor,
                CblasNoTrans, CblasTrans,
                M, N, K,
                alpha,
                A, K,          // lda = K
                B, B_cols,     // ldb = K (num columns of B as stored)
                beta,
                C, N);         // ldc = N
}


/* ---------- reusable GEMM (row-major) ---------- */

static inline void im2col_f32(const float *x, float *col, int N, int C, int H,
                              int W, int Kh, int Kw, int stride, int pad,
                              int OH, int OW)
{
  // 1) im2col: col[row, k] with row = ((n*OH + oy)*OW + ox)
  // k enumerates (c, ky, kx) in channel-major order:
  // k = c*(Kh*Kw) + ky*Kw + kx
  const int K    = C*Kh*Kw;

  // Not needed as we overwrite all values
  // const int rows = N*OH*OW;
  // memset(col, 0, (size_t)rows * (size_t)K * sizeof(float));

  int row = 0;
  for (int n = 0; n < N; n++) {
    for (int oy = 0; oy < OH; oy++) {
      const int base_y = oy * stride - pad;
      for (int ox = 0; ox < OW; ox++, row++) {
        const int base_x = ox * stride - pad;

        int k = 0;
        for (int c = 0; c < C; c++) {
          const int xc_base = (n * C + c) * H * W;
          for (int ky = 0; ky < Kh; ky++) {
            const int iy = base_y + ky;
            for (int kx = 0; kx < Kw; kx++, k++) {
              const int ix = base_x + kx;
              float v = 0.0f;
              if ((unsigned)iy < (unsigned)H && (unsigned)ix < (unsigned)W) {
                v = x[xc_base + iy * W + ix];
              }
              col[row * K + k] = v;
            }
          }
        }
      }
    }
  }
}

static inline void col2im_f32(const float *dcol, float *dx, int N, int C, int H,
                              int W, int Kh, int Kw, int stride, int pad,
                              int OH, int OW)
{
    memset(dx, 0, sizeof(float) * (size_t)N * (size_t)C * (size_t)H * (size_t)W);

    int cols = C * Kh * Kw;

    int row = 0;
    for (int n = 0; n < N; n++) {
        for (int oy = 0; oy < OH; oy++) {
            for (int ox = 0; ox < OW; ox++, row++) {
                int col = 0;
                for (int c = 0; c < C; c++) {
                    for (int ky = 0; ky < Kh; ky++) {
                        int iy = oy * stride + ky - pad;
                        for (int kx = 0; kx < Kw; kx++, col++) {
                            int ix = ox * stride + kx - pad;
                            if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
                                size_t idx = ((size_t)((n*C + c)*H + iy))*W + (size_t)ix;
                                dx[idx] += dcol[(size_t)row * (size_t)cols + (size_t)col];
                            }
                        }
                    }
                }
            }
        }
    }
}

/* ---------- conv2d_forward worker (runs w/o GVL) ---------- */

typedef struct {
    const float *x;      // [N,C,H,W]
    const float *w;      // [Cout,Cin,Kh,Kw]
    const float *bias;   // [Cout] or NULL

    float *col;          // [rows,K]
    float *out2d;        // [rows,Cout]
    float *out;          // [N,Cout,OH,OW]

    int N, C, H, W;
    int Cout, Cin, Kh, Kw;
    int stride, pad;
    int OH, OW;
    int rows, K;
} conv2d_fwd_args;

static void* conv2d_forward_nogvl(void *ptr)
{
    conv2d_fwd_args *a = (conv2d_fwd_args*)ptr;

    const int N   = a->N;
    const int C   = a->C;
    const int H   = a->H;
    const int W   = a->W;
    const int Cout = a->Cout;
    const int Kh  = a->Kh;
    const int Kw  = a->Kw;
    const int stride = a->stride;
    const int pad    = a->pad;
    const int OH  = a->OH;
    const int OW  = a->OW;
    const int rows = a->rows;
    const int K    = a->K;

    const float *x = a->x;
    const float *w = a->w;

    float *col  = a->col;
    float *out2d = a->out2d;
    float *out  = a->out;

    /* 1) im2col: col[rows, K] */
    im2col_f32(x, col, N, C, H, W, Kh, Kw, stride, pad, OH, OW);

    // 2) GEMM: out2d = col (rows,K) * w_flat^T
    // w is [Cout,Cin,Kh,Kw] contiguous => treat as w_flat [Cout,K] row-major
    sgemm_rowmajor_nt(rows, Cout, K, col, w, out2d, Cout, K);

    // 3) Scatter + bias into out[N,Cout,OH,OW] (NCHW contiguous)
    // out index: (((n*Cout + co)*OH + oy)*OW + ox)
    if (a->bias) {
        const float *b = a->bias;
        int r = 0;
        for (int n = 0; n < N; n++) {
            for (int oy = 0; oy < OH; oy++) {
                for (int ox = 0; ox < OW; ox++, r++) {
                    const int out_base = ((n * Cout) * OH + oy) * OW + ox;
                    for (int co = 0; co < Cout; co++) {
                        out[out_base + co * OH * OW] = out2d[r * Cout + co] + b[co];
                    }
                }
            }
        }
    } else {
        int r = 0;
        for (int n = 0; n < N; n++) {
            for (int oy = 0; oy < OH; oy++) {
                for (int ox = 0; ox < OW; ox++, r++) {
                    const int out_base = ((n * Cout) * OH + oy) * OW + ox;
                    for (int co = 0; co < Cout; co++) {
                        out[out_base + co * OH * OW] = out2d[r * Cout + co];
                    }
                }
            }
        }
    }

    return NULL;
}

/* ---------- Maurograd::Ext.conv2d_forward ---------- */

static VALUE mg_conv2d_forward(VALUE self,
                               VALUE vx,
                               VALUE vw,
                               VALUE vbias,   // nil ok
                               VALUE vstride,
                               VALUE vpad,
                               VALUE workspace)
{
    // Basic type checks
    if (!rb_obj_is_kind_of(vx, cNArray_)) rb_raise(rb_eTypeError, "x must be Numo::NArray");
    if (!rb_obj_is_kind_of(vw, cNArray_)) rb_raise(rb_eTypeError, "w must be Numo::NArray");
    if (!rb_obj_is_kind_of(vx, cSFloat_)) rb_raise(rb_eArgError, "x must be Numo::SFloat");
    if (!rb_obj_is_kind_of(vw, cSFloat_)) rb_raise(rb_eArgError, "w must be Numo::SFloat");
    if (vbias != Qnil && !rb_obj_is_kind_of(vbias, cSFloat_)) rb_raise(rb_eArgError, "bias must be Numo::SFloat or nil");

    narray_t *nx, *nw;
    GetNArray(vx, nx);
    GetNArray(vw, nw);

    // Dim checks
    if (nx->ndim != 4) rb_raise(rb_eArgError, "x must be 4D (NCHW)");
    if (nw->ndim != 4) rb_raise(rb_eArgError, "w must be 4D (Cout,Cin,Kh,Kw)");

    const int stride = NUM2INT(vstride);
    const int pad    = NUM2INT(vpad);

    // Shapes (size_t -> int with bounds assumption)
    const int N = sz_to_int(nx->shape[0], "N");
    const int C = sz_to_int(nx->shape[1], "C");
    const int H = sz_to_int(nx->shape[2], "H");
    const int W = sz_to_int(nx->shape[3], "W");

    const int Cout = sz_to_int(nw->shape[0], "Cout");
    const int Cin  = sz_to_int(nw->shape[1], "Cin");
    const int Kh   = sz_to_int(nw->shape[2], "Kh");
    const int Kw   = sz_to_int(nw->shape[3], "Kw");

    if (Cin != C) rb_raise(rb_eArgError, "channel mismatch: x.C=%d, w.Cin=%d", C, Cin);

    if (vbias != Qnil) {
        narray_t *nb;
        GetNArray(vbias, nb);
        if (nb->ndim != 1 || (int)nb->shape[0] != Cout) {
            rb_raise(rb_eArgError, "bias must have shape [Cout]");
        }
    }

    const int OH = (H + 2*pad - Kh) / stride + 1;
    const int OW = (W + 2*pad - Kw) / stride + 1;
    if (N <= 0 || C <= 0 || H <= 0 || W <= 0) rb_raise(rb_eArgError, "invalid input shape");
    if (OH <= 0 || OW <= 0) rb_raise(rb_eArgError, "invalid output shape (OH/OW <= 0)");

    if ((size_t)N * (size_t)OH * (size_t)OW > (size_t)INT_MAX)
      rb_raise(rb_eArgError, "rows too large");
    if ((size_t)C * (size_t)Kh * (size_t)Kw > (size_t)INT_MAX)
      rb_raise(rb_eArgError, "K too large");

    const int rows = N * OH * OW;
    const int K    = C * Kh * Kw;

    // Allocate output NArray [N,Cout,OH,OW]
    size_t out_shape[4];
    out_shape[0] = (size_t)N;
    out_shape[1] = (size_t)Cout;
    out_shape[2] = (size_t)OH;
    out_shape[3] = (size_t)OW;

    // Allocate temporary buffers as NArray too (so Ruby GC manages them)
    // col: [rows, K], out2d: [rows, Cout]
    // size_t col_shape[2]  = { (size_t)rows, (size_t)K };
    // size_t o2_shape[2]   = { (size_t)rows, (size_t)Cout };
    VALUE vcol = sfloat_new_2d(rows, K);
    VALUE vout2 = sfloat_new_2d(rows, Cout);
    VALUE vout = sfloat_new_4d(N, Cout, OH, OW);

    // Pointers (get them BEFORE releasing GVL)
    const float *x = (const float*)na_get_pointer_for_read(vx);
    const float *w = (const float*)na_get_pointer_for_read(vw);

    const float *b = NULL;
    if (vbias != Qnil) {
        narray_t *nb;
        GetNArray(vbias, nb);
        // bias expected [Cout] or [1,Cout]… (puoi rendere più permissivo dopo)
        b = (const float*)na_get_pointer_for_read(vbias);
    }

    float *col  = (float*)na_get_pointer_for_write(vcol);
    float *out2d = (float*)na_get_pointer_for_write(vout2);
    float *out  = (float*)na_get_pointer_for_write(vout);

    conv2d_fwd_args args;
    args.x = x; args.w = w; args.bias = b;
    args.col = col; args.out2d = out2d; args.out = out;
    args.N=N; args.C=C; args.H=H; args.W=W;
    args.Cout=Cout; args.Cin=Cin; args.Kh=Kh; args.Kw=Kw;
    args.stride=stride; args.pad=pad;
    args.OH=OH; args.OW=OW;
    args.rows=rows; args.K=K;

    /* We tell the GC not to touch vcol and vout2. Theoretically not needed, but
     * an extra check is never wrong. */
    RB_GC_GUARD(vcol);
    RB_GC_GUARD(vout2);
    RB_GC_GUARD(vout);

    /* Run without GVL */
    rb_thread_call_without_gvl(conv2d_forward_nogvl, &args, RUBY_UBF_IO, NULL);

    /* Save col in workspace */
    rb_hash_aset(workspace, ID2SYM(rb_intern("col")), vcol);

    return vout;
}


/* ---------- support functions for backward ---------- */

static inline void pack_dout2d_nchw(const float* dout, float* dout2d,
                                    int N, int Cout, int OH, int OW)
{
    int r = 0;
    for (int n = 0; n < N; n++) {
        for (int oy = 0; oy < OH; oy++) {
            for (int ox = 0; ox < OW; ox++, r++) {
                // dout layout: (((n*Cout + co)*OH + oy)*OW + ox)
                const int base = ((n * Cout) * OH + oy) * OW + ox;
                for (int co = 0; co < Cout; co++) {
                    dout2d[r * Cout + co] = dout[base + co * OH * OW];
                }
            }
        }
    }
}

static inline void reduce_dbias(const float* dout2d, float* db, int rows, int Cout)
{
    memset(db, 0, sizeof(float) * (size_t)Cout);
    for (int r = 0; r < rows; r++) {
        const float *p = dout2d + r * Cout;
        for (int co = 0; co < Cout; co++) db[co] += p[co];
    }
}

typedef struct {
    // inputs
    const float *x;     // [N,C,H,W]
    const float *w;     // [Cout,C,Kh,Kw] contiguous => treat as [Cout,K]
    const float *dout;  // [N,Cout,OH,OW]

    // temps
    float *dout2d; // [rows,Cout] (may be NULL if not needed)
    float *col;    // [rows,K]    (may be NULL if not needed)
    float *dcol;   // [rows,K]    (may be NULL if not needed)

    // outputs
    float *dx;     // [N,C,H,W] (may be NULL)
    float *dw;     // [Cout,K]  (may be NULL) but stored into [Cout,C,Kh,Kw] memory
    float *db;     // [Cout]    (may be NULL)

    // dims
    int N, C, H, W;
    int Cout, Kh, Kw;
    int OH, OW;
    int rows, K;
    int stride, pad;

    int want_dx, want_dw, want_db;

    bool have_col;

    // Non-finite guard: checked after each step, surfaced as a Ruby exception.
    int bad;        // 0 ok, 1 found non-finite
    int bad_where;  // 1=dout2d, 2=dw, 3=dcol, 4=dx, 5=db
} conv2d_bwd_args;


static void* conv2d_backward_nogvl(void *ptr)
{
    conv2d_bwd_args *a = (conv2d_bwd_args*)ptr;

    // Precondizione: se serve qualcosa, dout2d deve esistere
    if ((a->want_dx || a->want_dw || a->want_db) && !a->dout2d) {
        return NULL; // meglio sarebbe non arrivarci mai
    }


    // 1) pack dout -> dout2d
    pack_dout2d_nchw(a->dout, a->dout2d, a->N, a->Cout, a->OH, a->OW);

    if (a->dout2d) {
        // Full-array check; could be restricted to a subset for speed.
        if (any_nonfinite(a->dout2d, (size_t)a->rows * (size_t)a->Cout)) {
            a->bad = 1; a->bad_where = 1; return NULL;
        }
    }


    // 2) db
    if (a->want_db && a->db) {
        reduce_dbias(a->dout2d, a->db, a->rows, a->Cout);
    }

    if (a->want_db && a->db) {
        if (any_nonfinite(a->db, (size_t)a->Cout)) {
            a->bad = 1; a->bad_where = 5; return NULL;
        }
    }

    // 3) dw
    if (a->want_dw && a->dw) {
        // Per dw serve col
        if (a->col) {
            if (!a->have_col) {
                im2col_f32(a->x, a->col,
                           a->N, a->C, a->H, a->W,
                           a->Kh, a->Kw,
                           a->stride, a->pad,
                           a->OH, a->OW);
            }

            cblas_sgemm(CblasRowMajor,
                        CblasTrans, CblasNoTrans,
                        a->Cout, a->K, a->rows,
                        1.0f,
                        a->dout2d, a->Cout,
                        a->col, a->K,
                        0.0f,
                        a->dw, a->K);
        }
        // niente return: se col fosse NULL, meglio lasciare non calcolato,
        // ma in realtà NON dovrebbe accadere (vedi fix 2 sotto).
    }

    if (a->want_dw && a->dw) {
        if (any_nonfinite(a->dw, (size_t)a->Cout * (size_t)a->K)) {
            a->bad = 1; a->bad_where = 2; return NULL;
        }
    }

    // 4) dx
    if (a->want_dx && a->dx) {
        // Per dx NON serve col; serve dcol
        if (a->dcol) {
            cblas_sgemm(CblasRowMajor,
                        CblasNoTrans, CblasNoTrans,
                        a->rows, a->K, a->Cout,
                        1.0f,
                        a->dout2d, a->Cout,
                        a->w, a->K,
                        0.0f,
                        a->dcol, a->K);

            if (a->want_dx && a->dcol) {
                if (any_nonfinite(a->dcol, (size_t)a->rows * (size_t)a->K)) {
                    a->bad = 1; a->bad_where = 3; return NULL;
                }
            }

            col2im_f32(a->dcol, a->dx,
                       a->N, a->C, a->H, a->W,
                       a->Kh, a->Kw,
                       a->stride, a->pad,
                       a->OH, a->OW);

            if (a->want_dx && a->dx) {
                if (any_nonfinite(a->dx, (size_t)a->N * (size_t)a->C * (size_t)a->H * (size_t)a->W)) {
                    a->bad = 1; a->bad_where = 4; return NULL;
                }
            }
        }
        // niente return
    }



    return NULL;
}



/*
static void* conv2d_backward_nogvl(void *ptr)
{
    conv2d_bwd_args *a = (conv2d_bwd_args*)ptr;

    if ((a->want_dx || a->want_dw || a->want_db) && !a->dout2d) return NULL;

    // 1) dout2d
    if (a->dout2d) {
        pack_dout2d_nchw(a->dout, a->dout2d, a->N, a->Cout, a->OH, a->OW);
    }

    // 2) db
    if (a->want_db && a->db) {
        reduce_dbias(a->dout2d, a->db, a->rows, a->Cout);
    }

    // 3) dw needs col
    if (a->want_dw && a->dw) {
        if (!a->col) return NULL;
                                  //
        if (a->have_col == false) {
            // need to compute col
            // col = im2col(x)
            im2col_f32(a->x, a->col,
                       a->N, a->C, a->H, a->W,
                       a->Kh, a->Kw,
                       a->stride, a->pad,
                       a->OH, a->OW);
        }

        if (!a->dout2d) return NULL;

        // dw(Cout,K) = dout2d^T(Cout,rows) * col(rows,K)
        cblas_sgemm(CblasRowMajor,
                    CblasTrans, CblasNoTrans,
                    a->Cout, a->K, a->rows,
                    1.0f,
                    a->dout2d, a->Cout,  // A is [rows,Cout], trans => [Cout,rows], lda = Cout
                    a->col, a->K,        // B is [rows,K], ldb = K
                    0.0f,
                    a->dw, a->K);        // C is [Cout,K], ldc = K
    }

    // 4) dx needs dcol
    if (a->want_dx && a->dx) {
        if (!a->col) return NULL;

        // dcol(rows,K) = dout2d(rows,Cout) * w(Cout,K)
        cblas_sgemm(CblasRowMajor,
                    CblasNoTrans, CblasNoTrans,
                    a->rows, a->K, a->Cout,
                    1.0f,
                    a->dout2d, a->Cout, // [rows,Cout]
                    a->w, a->K,         // [Cout,K]
                    0.0f,
                    a->dcol, a->K);     // [rows,K]

        // dx = col2im(dcol)
        col2im_f32(a->dcol, a->dx,
                   a->N, a->C, a->H, a->W,
                   a->Kh, a->Kw,
                   a->stride, a->pad,
                   a->OH, a->OW);
    }

    return NULL;
}
*/


/* ---------- Maurograd::Ext.conv2d_backward ---------- */

static VALUE mg_conv2d_backward(VALUE self, VALUE vx, VALUE vw, VALUE vdout,
                                VALUE vstride, VALUE vpad,
                                VALUE vwant_dx, VALUE vwant_dw, VALUE vwant_db,
                                VALUE workspace)
{
    // type checks
    if (!rb_obj_is_kind_of(vx, cNArray_) || !rb_obj_is_kind_of(vx, cSFloat_))
        rb_raise(rb_eTypeError, "x must be Numo::SFloat");
    if (!rb_obj_is_kind_of(vw, cNArray_) || !rb_obj_is_kind_of(vw, cSFloat_))
        rb_raise(rb_eTypeError, "w must be Numo::SFloat");
    if (!rb_obj_is_kind_of(vdout, cNArray_) || !rb_obj_is_kind_of(vdout, cSFloat_))
        rb_raise(rb_eTypeError, "dout must be Numo::SFloat");

    narray_t *nx, *nw, *ndout;
    GetNArray(vx, nx);
    GetNArray(vw, nw);
    GetNArray(vdout, ndout);

    if (nx->ndim != 4)   rb_raise(rb_eArgError, "x must be 4D [N,C,H,W]");
    if (nw->ndim != 4)   rb_raise(rb_eArgError, "w must be 4D [Cout,C,Kh,Kw]");
    if (ndout->ndim != 4)rb_raise(rb_eArgError, "dout must be 4D [N,Cout,OH,OW]");

    const int stride = NUM2INT(vstride);
    const int pad    = NUM2INT(vpad);

    const int want_dx = RTEST(vwant_dx) ? 1 : 0;
    const int want_dw = RTEST(vwant_dw) ? 1 : 0;
    const int want_db = RTEST(vwant_db) ? 1 : 0;

    // shapes
    const int N   = sz_to_int(nx->shape[0], "N");
    const int C   = sz_to_int(nx->shape[1], "C");
    const int H   = sz_to_int(nx->shape[2], "H");
    const int W   = sz_to_int(nx->shape[3], "W");

    const int Cout = sz_to_int(nw->shape[0], "Cout");
    const int Cin  = sz_to_int(nw->shape[1], "Cin");
    const int Kh   = sz_to_int(nw->shape[2], "Kh");
    const int Kw   = sz_to_int(nw->shape[3], "Kw");

    if (Cin != C) rb_raise(rb_eArgError, "channel mismatch: x.C=%d, w.Cin=%d", C, Cin);

    const int dN    = sz_to_int(ndout->shape[0], "dout.N");
    const int dCout = sz_to_int(ndout->shape[1], "dout.Cout");
    const int OH    = sz_to_int(ndout->shape[2], "OH");
    const int OW    = sz_to_int(ndout->shape[3], "OW");

    if (dN != N)       rb_raise(rb_eArgError, "dout N mismatch: %d vs %d", dN, N);
    if (dCout != Cout) rb_raise(rb_eArgError, "dout Cout mismatch: %d vs %d", dCout, Cout);

    // expected OH/OW sanity
    const int exp_OH = (H + 2*pad - Kh) / stride + 1;
    const int exp_OW = (W + 2*pad - Kw) / stride + 1;
    if (OH != exp_OH || OW != exp_OW) {
        rb_raise(rb_eArgError, "dout OH/OW mismatch: got (%d,%d), expected (%d,%d)",
                 OH, OW, exp_OH, exp_OW);
    }

    const int rows = N * OH * OW;
    const int K    = C * Kh * Kw;

    // Allocate outputs (or nil)
    VALUE vdx = Qnil, vdw = Qnil, vdb = Qnil;

    if (want_dx) {
        vdx = sfloat_new_4d(N, C, H, W);
        float *dx = (float*)na_get_pointer_for_write(vdx);
        memset(dx, 0, sizeof(float) * (size_t)N * C * H * W);
    }
    if (want_dw) {
        vdw = sfloat_new_4d(Cout, C, Kh, Kw);
        float *dwp = (float*)na_get_pointer_for_write(vdw);
        memset(dwp, 0, sizeof(float) * (size_t)Cout * (size_t)C * (size_t)Kh * (size_t)Kw);
    }
    if (want_db) {
        size_t args[1] = { (size_t)(Cout) };
        vdb = nary_new(cSFloat_, 1, args);
        float *dbp = (float*)na_get_pointer_for_write(vdb);
        memset(dbp, 0, sizeof(float) * (size_t)Cout);
    }

    // Allocate temporaries as needed
    VALUE vdout2d = Qnil;
    VALUE vdcol   = Qnil;

    if (want_dx || want_dw || want_db) {
        vdout2d = sfloat_new_2d(rows, Cout);
        float *dout2d = (float*)na_get_pointer_for_write(vdout2d);
        memset(dout2d, 0, sizeof(float) * (size_t)(rows * Cout));
    }
    if (want_dx) {
        vdcol = sfloat_new_2d(rows, K);
        float *dcol = (float*)na_get_pointer_for_write(vdcol);
        memset(dcol, 0, sizeof(float) * (size_t)(rows * K));
    }

    // Get pointers BEFORE releasing GVL
    const float *x    = (const float*)na_get_pointer_for_read(vx);
    const float *w    = (const float*)na_get_pointer_for_read(vw);
    const float *dout = (const float*)na_get_pointer_for_read(vdout);

    float *dout2d = (vdout2d == Qnil) ? NULL : (float*)na_get_pointer_for_write(vdout2d);
    float *dcol   = (vdcol   == Qnil) ? NULL : (float*)na_get_pointer_for_write(vdcol);

    float *dx = (vdx == Qnil) ? NULL : (float*)na_get_pointer_for_write(vdx);
    float *dw = (vdw == Qnil) ? NULL : (float*)na_get_pointer_for_write(vdw);
    float *db = (vdb == Qnil) ? NULL : (float*)na_get_pointer_for_write(vdb);

    VALUE ws_col = rb_hash_aref(workspace, sym_col);

    int have_col = 0;
    if (ws_col != Qnil) {
        // Deve essere SFloat 2D [rows, K]
        if (rb_obj_is_kind_of(ws_col, cNArray_) &&
            rb_obj_is_kind_of(ws_col, cSFloat_)) {
            narray_t *nws;
            GetNArray(ws_col, nws);
            if (nws->ndim == 2) {
                int r0 = sz_to_int(nws->shape[0], "ws_col.rows");
                int k0 = sz_to_int(nws->shape[1], "ws_col.K");
                if (r0 == rows && k0 == K) {
                    have_col = 1;
                }
            }
        }
    }

    VALUE vcol = Qnil;
    if (want_dw && !have_col) {
        /* Need to allocate col */
        vcol = sfloat_new_2d(rows, K);
    } else if (want_dw && have_col) {
        /* Use col from workspace */
        vcol = ws_col;
    }

    float *col = NULL;
    if (vcol != Qnil) {
        if (have_col) {
            /* We take col from workspace, so we need the read pointer */
            col = (float*)na_get_pointer_for_read(vcol);
        } else {
            /* We have to calculate col with im2col */
            col = (float*)na_get_pointer_for_write(vcol);
        }
    }

    conv2d_bwd_args args;
    args.x = x; args.w = w; args.dout = dout;
    args.dout2d = dout2d;
    args.dcol = dcol;
    args.dx = dx;
    args.dw = dw;
    args.db = db;
    args.N=N; args.C=C; args.H=H; args.W=W;
    args.Cout=Cout; args.Kh=Kh; args.Kw=Kw;
    args.OH=OH; args.OW=OW;
    args.rows=rows; args.K=K;
    args.stride=stride; args.pad=pad;
    args.want_dx=want_dx; args.want_dw=want_dw; args.want_db=want_db;
    args.col = col;
    args.have_col = have_col;

    args.bad = 0;
    args.bad_where = 0;

    /* Guard temporary and output variables */
    RB_GC_GUARD(vdx);
    RB_GC_GUARD(vdw);
    RB_GC_GUARD(vdb);
    RB_GC_GUARD(vdout2d);
    RB_GC_GUARD(vcol);
    RB_GC_GUARD(vdcol);

    if ((want_dx || want_dw || want_db) && !dout2d) {
        rb_raise(rb_eRuntimeError, "internal: dout2d is NULL but required");
    }
    if (want_dx && (!dx || !dcol)) {
        rb_raise(rb_eRuntimeError, "internal: dx/dcol is NULL but want_dx=1");
    }
    if (want_dw && (!dw || !col)) {
        rb_raise(rb_eRuntimeError, "internal: dw/col is NULL but want_dw=1");
    }
    if (want_db && (!db)) {
        rb_raise(rb_eRuntimeError, "internal: db is NULL but want_db=1");
    }

    /* Run without GVL */
    rb_thread_call_without_gvl(conv2d_backward_nogvl, &args, RUBY_UBF_IO, NULL);

    if (args.bad) {
        const char *where =
            (args.bad_where==1) ? "dout2d" :
            (args.bad_where==2) ? "dw" :
            (args.bad_where==3) ? "dcol" :
            (args.bad_where==4) ? "dx" :
            (args.bad_where==5) ? "db" : "unknown";
        rb_raise(rb_eRuntimeError, "conv2d_backward produced NaN/Inf in %s", where);
    }


    return rb_ary_new_from_args(3, vdx, vdw, vdb);
}


/* ---------- Maurograd::Ext.im2col ---------- */

static VALUE mg_im2col(VALUE self, VALUE vx, VALUE vkh, VALUE vkw, VALUE vstride, VALUE vpad)
{
    narray_t *nx;
    GetNArray(vx, nx);

    if (!rb_obj_is_kind_of(vx, cSFloat_)) {
        rb_raise(rb_eTypeError, "im2col supports only Numo::SFloat");
    }

    int kh = NUM2INT(vkh), kw = NUM2INT(vkw);
    int stride = NUM2INT(vstride), pad = NUM2INT(vpad);

    if (nx->ndim != 4) rb_raise(rb_eArgError, "x must be 4D (NCHW)");

    int N = sz_to_int(nx->shape[0], "N");
    int C = sz_to_int(nx->shape[1], "C");
    int H = sz_to_int(nx->shape[2], "H");
    int W = sz_to_int(nx->shape[3], "W");

    int OH = (H + 2*pad - kh) / stride + 1;
    int OW = (W + 2*pad - kw) / stride + 1;
    if (OH <= 0 || OW <= 0) rb_raise(rb_eArgError, "invalid output shape");

    int rows = N * OH * OW;
    int cols = C * kh * kw;

    size_t out_shape[2] = { (size_t)rows, (size_t)cols };
    VALUE vout = nary_new(cSFloat_, 2, out_shape);

    float *x   = (float*)na_get_pointer_for_read(vx);
    float *out = (float*)na_get_pointer_for_write(vout);

    im2col_f32(x, out, N, C, H, W, kh, kw, stride, pad, OH, OW);

    return vout;
}



/* ---------- col2im ---------- */

static VALUE mg_col2im(VALUE self,
                       VALUE vdcol,
                       VALUE vxshape,
                       VALUE vkh,
                       VALUE vkw,
                       VALUE vstride,
                       VALUE vpad)
{
    narray_t *ndc;
    GetNArray(vdcol, ndc);

    if (!rb_obj_is_kind_of(vdcol, cSFloat_)) {
        rb_raise(rb_eTypeError, "col2im supports only Numo::SFloat");
    }

    int Kh = NUM2INT(vkh);
    int Kw = NUM2INT(vkw);
    int stride = NUM2INT(vstride);
    int pad = NUM2INT(vpad);

    int N = sz_to_int(NUM2SIZET(rb_ary_entry(vxshape, 0)), "N");
    int C = sz_to_int(NUM2SIZET(rb_ary_entry(vxshape, 1)), "C");
    int H = sz_to_int(NUM2SIZET(rb_ary_entry(vxshape, 2)), "H");
    int W = sz_to_int(NUM2SIZET(rb_ary_entry(vxshape, 3)), "W");

    int OH = (H + 2*pad - Kh) / stride + 1;
    int OW = (W + 2*pad - Kw) / stride + 1;
    if (OH <= 0 || OW <= 0) rb_raise(rb_eArgError, "invalid output shape");

    size_t out_shape[4] = { (size_t)N, (size_t)C, (size_t)H, (size_t)W };
    VALUE vdx = nary_new(cSFloat_, 4, out_shape);

    float *dx   = (float*)na_get_pointer_for_write(vdx);
    float *dcol = (float*)na_get_pointer_for_read(vdcol);

    col2im_f32(dcol, dx, N, C, H, W, Kh, Kw, stride, pad, OH, OW);

    return vdx;
}



/* ---------- Init ---------- */

void Init_maurograd_ext(void)
{
    cNumo   = rb_const_get(rb_cObject, rb_intern("Numo"));
    cNArray_ = rb_const_get(cNumo, rb_intern("NArray"));
    cSFloat_ = rb_const_get(cNumo, rb_intern("SFloat"));
    id_zeros = rb_intern("zeros");

    mMaurograd = rb_define_module("Maurograd");
    mExt = rb_define_module_under(mMaurograd, "Ext");

    rb_define_singleton_method(mExt, "conv2d_forward", mg_conv2d_forward, 6);
    rb_define_singleton_method(mExt, "conv2d_backward", mg_conv2d_backward, 9);

    rb_define_singleton_method(mExt, "im2col", mg_im2col, 5);
    rb_define_singleton_method(mExt, "col2im", mg_col2im, 6);

    id_col = rb_intern("col");
    sym_col = ID2SYM(id_col);
    rb_global_variable(&sym_col); // optional, but good for safety

    openblas_set_num_threads(1);
}

/*
 * vim: set ts=4 sw=4 et ai
 */
