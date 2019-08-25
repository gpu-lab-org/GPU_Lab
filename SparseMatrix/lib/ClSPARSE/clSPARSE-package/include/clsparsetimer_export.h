
#ifndef CLSPARSETIMER_EXPORT_H
#define CLSPARSETIMER_EXPORT_H

#ifdef CLSPARSETIMER_STATIC_DEFINE
#  define CLSPARSETIMER_EXPORT
#  define CLSPARSETIMER_NO_EXPORT
#else
#  ifndef CLSPARSETIMER_EXPORT
#    ifdef clsparseTimer_EXPORTS
        /* We are building this library */
#      define CLSPARSETIMER_EXPORT __attribute__((visibility("default")))
#    else
        /* We are using this library */
#      define CLSPARSETIMER_EXPORT __attribute__((visibility("default")))
#    endif
#  endif

#  ifndef CLSPARSETIMER_NO_EXPORT
#    define CLSPARSETIMER_NO_EXPORT __attribute__((visibility("hidden")))
#  endif
#endif

#ifndef CLSPARSETIMER_DEPRECATED
#  define CLSPARSETIMER_DEPRECATED __attribute__ ((__deprecated__))
#endif

#ifndef CLSPARSETIMER_DEPRECATED_EXPORT
#  define CLSPARSETIMER_DEPRECATED_EXPORT CLSPARSETIMER_EXPORT CLSPARSETIMER_DEPRECATED
#endif

#ifndef CLSPARSETIMER_DEPRECATED_NO_EXPORT
#  define CLSPARSETIMER_DEPRECATED_NO_EXPORT CLSPARSETIMER_NO_EXPORT CLSPARSETIMER_DEPRECATED
#endif

#define DEFINE_NO_DEPRECATED 0
#if DEFINE_NO_DEPRECATED
# define CLSPARSETIMER_NO_DEPRECATED
#endif

#endif
