# This file was automatically generated by SWIG (http://www.swig.org).
# Version 2.0.4
#
# Do not make changes to this file unless you know what you are doing--modify
# the SWIG interface file instead.



from sys import version_info
if version_info >= (2,6,0):
    def swig_import_helper():
        from os.path import dirname
        import imp
        fp = None
        try:
            fp, pathname, description = imp.find_module('_truncated_normal', [dirname(__file__)])
        except ImportError:
            import _truncated_normal
            return _truncated_normal
        if fp is not None:
            try:
                _mod = imp.load_module('_truncated_normal', fp, pathname, description)
            finally:
                fp.close()
            return _mod
    _truncated_normal = swig_import_helper()
    del swig_import_helper
else:
    import _truncated_normal
del version_info
try:
    _swig_property = property
except NameError:
    pass # Python < 2.2 doesn't have 'property'.
def _swig_setattr_nondynamic(self,class_type,name,value,static=1):
    if (name == "thisown"): return self.this.own(value)
    if (name == "this"):
        if type(value).__name__ == 'SwigPyObject':
            self.__dict__[name] = value
            return
    method = class_type.__swig_setmethods__.get(name,None)
    if method: return method(self,value)
    if (not static):
        self.__dict__[name] = value
    else:
        raise AttributeError("You cannot add attributes to %s" % self)

def _swig_setattr(self,class_type,name,value):
    return _swig_setattr_nondynamic(self,class_type,name,value,0)

def _swig_getattr(self,class_type,name):
    if (name == "thisown"): return self.this.own()
    method = class_type.__swig_getmethods__.get(name,None)
    if method: return method(self)
    raise AttributeError(name)

def _swig_repr(self):
    try: strthis = "proxy of " + self.this.__repr__()
    except: strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)

try:
    _object = object
    _newclass = 1
except AttributeError:
    class _object : pass
    _newclass = 0



def normal_cdf(*args):
  return _truncated_normal.normal_cdf(*args)
normal_cdf = _truncated_normal.normal_cdf

def normal_cdf_array(*args):
  return _truncated_normal.normal_cdf_array(*args)
normal_cdf_array = _truncated_normal.normal_cdf_array

def normal_pdf(*args):
  return _truncated_normal.normal_pdf(*args)
normal_pdf = _truncated_normal.normal_pdf

def normal_pdf_array(*args):
  return _truncated_normal.normal_pdf_array(*args)
normal_pdf_array = _truncated_normal.normal_pdf_array

def normal_cdf_inv(*args):
  return _truncated_normal.normal_cdf_inv(*args)
normal_cdf_inv = _truncated_normal.normal_cdf_inv

def truncated_normal_ab_cdf(*args):
  return _truncated_normal.truncated_normal_ab_cdf(*args)
truncated_normal_ab_cdf = _truncated_normal.truncated_normal_ab_cdf

def truncated_normal_ab_cdf_array(*args):
  return _truncated_normal.truncated_normal_ab_cdf_array(*args)
truncated_normal_ab_cdf_array = _truncated_normal.truncated_normal_ab_cdf_array

def truncated_normal_ab_pdf(*args):
  return _truncated_normal.truncated_normal_ab_pdf(*args)
truncated_normal_ab_pdf = _truncated_normal.truncated_normal_ab_pdf

def truncated_normal_ab_pdf_array(*args):
  return _truncated_normal.truncated_normal_ab_pdf_array(*args)
truncated_normal_ab_pdf_array = _truncated_normal.truncated_normal_ab_pdf_array

def truncated_normal_ab_cdf_inv(*args):
  return _truncated_normal.truncated_normal_ab_cdf_inv(*args)
truncated_normal_ab_cdf_inv = _truncated_normal.truncated_normal_ab_cdf_inv
# This file is compatible with both classic and new-style classes.


