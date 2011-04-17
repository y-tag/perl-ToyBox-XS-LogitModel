#ifdef __cplusplus
extern "C" {
#endif
#include "EXTERN.h"
#include "perl.h"
#include "XSUB.h"
#ifdef __cplusplus
}
#endif

#include "ppport.h"
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_vector.h>
#include <string>
#include <vector>
#include <map>
#include <cfloat>
#include <cmath>

typedef std::map<int, int> IntToIntMap;
typedef std::map<int, double> IntToDoubleMap;
typedef std::map<std::string, int> StrToIntMap;
typedef std::map<std::string, double> StrToDoubleMap;
typedef std::map<std::string, std::map<std::string, double> > Str2ToDoubleMap;

typedef std::vector<IntToDoubleMap> IDMapVector;

typedef struct{
  IDMapVector data;
  std::vector<int> labels;
  int fnum;
  int lnum;
  double sigma;
} param_struct;

class LogitModel{
  public:
    LogitModel();
    ~LogitModel();
    void AddInstance(const StrToIntMap &doc, const std::string &label);
    void Train(const size_t max_iterations, const std::string &algorithm, const double sigma);
    StrToDoubleMap Predict(const StrToIntMap &doc);
  private:
    int numData;
    StrToIntMap fdict;
    StrToIntMap ldict;
    IDMapVector data;
    std::vector<int> labels;
    std::vector<double> weight;
};

double my_f(const gsl_vector *v, void *params);
void my_df(const gsl_vector *v, void *params, gsl_vector *df);
void my_fdf (const gsl_vector *x, void *params, double *f, gsl_vector *df);


LogitModel::LogitModel()
  : numData(0), fdict(), data(), labels(), weight()
{
}

LogitModel::~LogitModel()
{
}

void
LogitModel::AddInstance(const StrToIntMap &doc, const std::string &label)
{

  IntToDoubleMap int_doc;

  for (StrToIntMap::const_iterator it = doc.begin(); it != doc.end(); ++it) {
    if (fdict.find(it->first) == fdict.end()) {
      int fnum = fdict.size();
      fdict[it->first] = fnum;
    }
    int feature = fdict[it->first];
    int_doc[feature] = it->second;
  }

  if (ldict.find(label) == ldict.end()) {
    int lnum = ldict.size();
    ldict[label] = lnum;
  }

  data.push_back(int_doc);
  labels.push_back(ldict[label]);
  ++numData;

  return;
}

void
LogitModel::Train(const size_t max_iterations, const std::string &algorithm, const double sigma)
{
  double step_size = 1e-2;
  double tol = 1e-4;
  double epsabs = 1e-3;

  size_t iter = 0;
  int status;

  param_struct par;
  par.data   = data;
  par.labels = labels;
  par.fnum   = fdict.size();
  par.lnum   = ldict.size();
  par.sigma  = sigma;

  int weight_size = fdict.size() * (ldict.size() - 1);

  gsl_vector *x;
  x = gsl_vector_alloc(weight_size);
  for (int i = 0; i < weight_size; ++i) {
    gsl_vector_set(x, i, 0.0);
  }

  gsl_multimin_function_fdf my_func;

  my_func.n = weight_size;
  my_func.f = &my_f;
  my_func.df = &my_df;
  my_func.fdf = &my_fdf;
  my_func.params = &par;

  const gsl_multimin_fdfminimizer_type *T;
  if (algorithm == "conjugate_fr") {
    T = gsl_multimin_fdfminimizer_conjugate_fr;
  } else if (algorithm == "conjugate_pr") {
    T = gsl_multimin_fdfminimizer_conjugate_pr;
  } else if (algorithm == "steepest_descent") {
    T = gsl_multimin_fdfminimizer_steepest_descent;
  } else {
    T = gsl_multimin_fdfminimizer_vector_bfgs2;
  }

  gsl_multimin_fdfminimizer *s;
  s = gsl_multimin_fdfminimizer_alloc(T, weight_size);

  gsl_multimin_fdfminimizer_set(s, &my_func, x, step_size, tol);


  do
  {
    iter++;
    status = gsl_multimin_fdfminimizer_iterate(s);

    if (status) {
      fprintf(stderr, "status: %d, GSL_ENOPROG: %d\n", status, GSL_ENOPROG);
      break;
    }

    status = gsl_multimin_test_gradient(s->gradient, epsabs);

    if (status == GSL_SUCCESS) {
      fprintf(stderr, "Minimum found at:\n");
    }

    fprintf(stderr, "%5d %10.16f\n", iter, s->f);

  }
  while (status == GSL_CONTINUE && iter < max_iterations);


  for (int i = 0; i < (s->x)->size; ++i) {
    weight.push_back(gsl_vector_get(s->x, i));
  }

  gsl_multimin_fdfminimizer_free(s);
  gsl_vector_free(x);

  return;
}

StrToDoubleMap
LogitModel::Predict(const StrToIntMap &doc)
{

  StrToDoubleMap result;
  std::vector<double> exp_sum;
  double denom = 0.0;

  int fnum = fdict.size();
  int lnum = ldict.size();

  // for 1, ..., L-1
  for (int i = 0; i < lnum - 1; ++i) {
    double tmp_sum = 0.0;

    for (StrToIntMap::const_iterator fit = doc.begin(); fit != doc.end(); ++fit) {
      if (fdict.find(fit->first) == fdict.end()) {
        continue;
      }
      int feature = fnum * i + fdict[fit->first];
      tmp_sum += weight[feature] * fit->second;
    }
    tmp_sum = exp(tmp_sum);
    denom += tmp_sum;
    exp_sum.push_back(tmp_sum);
  }

  // for L
  denom += 1.0;
  exp_sum.push_back(1.0);

  for (StrToIntMap::iterator lit = ldict.begin(); lit != ldict.end(); ++lit) {
    result[lit->first] = exp_sum[lit->second] / denom;
  }

  return result;
}


double
my_f(const gsl_vector *v, void *params)
{
  param_struct *par = static_cast<param_struct*>(params);
  IDMapVector data = par->data;
  std::vector<int> labels = par->labels;
  int fnum = par->fnum;
  int lnum = par->lnum;
  double sigma = par->sigma;

  double f = 0.0;

  for (int i = 0; i < data.size(); ++i) {
    IntToDoubleMap datum = data[i];
    int label = labels[i];

    double denom = 0.0;

    for (int j = 0; j < lnum - 1; ++j) {
      double linear_sum = 0.0;

      for (IntToDoubleMap::const_iterator it = datum.begin(); it != datum.end(); ++it) {
        int feature = fnum * j + it->first;
        linear_sum += gsl_vector_get(v, feature) * it->second;
      }

      double exp_sum = exp(linear_sum);
      denom += exp_sum;

      if (label == j) {
        f -= linear_sum;
      }
    }

    denom += 1.0;

    f += log(denom);
  }

  for (int i = 0; i < v->size; ++i) {
    double tmp = gsl_vector_get(v, i);
    f += (tmp * tmp) / (2 * sigma * sigma);
  }

  return f;
}

void
my_df(const gsl_vector *v, void *params, gsl_vector *df)
{
  param_struct *par = static_cast<param_struct*>(params);
  IDMapVector data = par->data;
  std::vector<int> labels = par->labels;
  int fnum = par->fnum;
  int lnum = par->lnum;
  double sigma = par->sigma;

  for (int i = 0; i < df->size; ++i) {
    gsl_vector_set(df, i, 0.0);
  }

  for (int i = 0; i < data.size(); ++i) {
    IntToDoubleMap datum = data[i];
    int label = labels[i];

    std::vector<double> numer;
    double denom = 0.0;

    for (int j = 0; j < lnum - 1; ++j) {
      double linear_sum = 0.0;

      for (IntToDoubleMap::const_iterator it = datum.begin(); it != datum.end(); ++it) {
        int feature = fnum * j + it->first;
        linear_sum += gsl_vector_get(v, feature) * it->second;
      }

      double exp_sum = exp(linear_sum);
      numer.push_back(exp_sum);
      denom += exp_sum;
    }

    denom += 1.0;

    for (int j = 0; j < lnum - 1; ++j) {
      double y = (label == j) ? 1.0 : 0.0;
      double diff = y - (numer[j] / denom);

      for (IntToDoubleMap::const_iterator it = datum.begin(); it != datum.end(); ++it) {
        int feature = fnum * j + it->first;
        double tmp = gsl_vector_get(df, feature) - diff * it->second;
        gsl_vector_set(df, feature, tmp);
      }
    }
  }

  for (int i = 0; i < v->size; ++i) {
    double tmp = gsl_vector_get(df, i) + (gsl_vector_get(v, i) / (sigma * sigma));
    gsl_vector_set(df, i, tmp);
  }

  return;
}

void
my_fdf (const gsl_vector *x, void *params, double *f, gsl_vector *df) 
{
  *f = my_f(x, params); 
  my_df(x, params, df);
} 


MODULE = ToyBox::XS::LogitModel		PACKAGE = ToyBox::XS::LogitModel	

LogitModel *
LogitModel::new()

void
LogitModel::DESTROY()

void
LogitModel::xs_add_instance(attributes_input, label_input)
  SV * attributes_input
  char* label_input
CODE:
  {
    HV *hv_attributes = (HV*) SvRV(attributes_input);
    SV *val;
    char *key;
    I32 retlen;
    int num = hv_iterinit(hv_attributes);
    std::string label = std::string(label_input);
    StrToIntMap attributes;

    for (int i = 0; i < num; ++i) {
      val = hv_iternextsv(hv_attributes, &key, &retlen);
      attributes[key] = (int)SvIV(val);
    }

    THIS->AddInstance(attributes, label);
  }

void
LogitModel::xs_train(max_iterations, algorithm, sigma)
  size_t max_iterations
  char* algorithm
  double sigma
CODE:
  {
    THIS->Train(max_iterations, std::string(algorithm), sigma);
  }

SV*
LogitModel::xs_predict(attributes_input)
  SV * attributes_input
CODE:
  {
    HV *hv_attributes = (HV*) SvRV(attributes_input);
    SV *val;
    char *key;
    I32 retlen;
    int num = hv_iterinit(hv_attributes);
    StrToIntMap attributes;
    StrToDoubleMap result;

    for (int i = 0; i < num; ++i) {
      val = hv_iternextsv(hv_attributes, &key, &retlen);
      attributes[key] = (int)SvIV(val);
    }

    result = THIS->Predict(attributes);

    HV *hv_result = newHV();
    for (StrToDoubleMap::iterator it = result.begin(); it != result.end(); ++it) {
      const char *const_key = (it->first).c_str();
      SV* val = newSVnv(it->second);
      hv_store(hv_result, const_key, strlen(const_key), val, 0); 
    }

    RETVAL = newRV_inc((SV*) hv_result);
  }
OUTPUT:
  RETVAL
  
