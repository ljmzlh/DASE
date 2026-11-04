/*
 * The conda PostgreSQL build was configured against a glibc that still had
 * <xlocale.h> (pg_config.h sets HAVE_XLOCALE_H); the conda sysroot used to
 * compile this module no longer ships it. Its contents moved into <locale.h>.
 */
#include <locale.h>
