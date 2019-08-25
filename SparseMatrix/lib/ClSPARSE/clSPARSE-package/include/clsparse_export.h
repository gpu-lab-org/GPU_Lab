
#ifndef CLSPARSE_EXPORT_H
#define CLSPARSE_EXPORT_H

#ifdef CLSPARSE_STATIC_DEFINE
#  define CLSPARSE_EXPORT
#  define CLSPARSE_NO_EXPORT
#else
#  ifndef CLSPARSE_EXPORT
#    ifdef clSPARSE_EXPORTS
        /* We are building this library */
#      define CLSPARSE_EXPORT __attribute__((visibility("default")))
#    else
        /* We are using this library */
#      define CLSPARSE_EXPORT __attribute__((visibility("default")))
#    endif
#  endif

#  ifndef CLSPARSE_NO_EXPORT
#    define CLSPARSE_NO_EXPORT __attribute__((visibility("hidden")))
#  endif
#endif

#ifndef CLSPARSE_DEPRECATED
#  define CLSPARSE_DEPRECATED __attribute__ ((__deprecated__))
#endif

#ifndef CLSPARSE_DEPRECATED_EXPORT
#  define CLSPARSE_DEPRECATED_EXPORT CLSPARSE_EXPORT CLSPARSE_DEPRECATED
#endif

#ifndef CLSPARSE_DEPRECATED_NO_EXPORT
#  define CLSPARSE_DEPRECATED_NO_EXPORT CLSPARSE_NO_EXPORT CLSPARSE_DEPRECATED
#endif

#define DEFINE_NO_DEPRECATED 0
#if DEFINE_NO_DEPRECATED
# define CLSPARSE_NO_DEPRECATED
#endif

#endif
