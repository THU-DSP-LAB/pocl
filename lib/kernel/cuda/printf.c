/* OpenCL built-in library: printf() for CUDA

   Copyright (c) 2016 James Price / University of Bristol
   Copyright (c) 2026 PoCL developers

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
*/

#include <stdarg.h>

#define POCL_CUDA_PRINTF_ARG_BYTES 128
#define POCL_CUDA_PRINTF_FORMAT_BYTES 32
#define POCL_CUDA_PRINTF_MAX_VECTOR_WIDTH 16

#define GENERIC_AS __attribute__ ((address_space (4)))

typedef GENERIC_AS const char *pocl_cuda_format_ptr;

typedef enum
{
  POCL_CUDA_PRINTF_LENGTH_DEFAULT,
  POCL_CUDA_PRINTF_LENGTH_HH,
  POCL_CUDA_PRINTF_LENGTH_H,
  POCL_CUDA_PRINTF_LENGTH_HL,
  POCL_CUDA_PRINTF_LENGTH_L
} pocl_cuda_printf_length;

typedef struct
{
  pocl_cuda_format_ptr next;
  char scalar_format[POCL_CUDA_PRINTF_FORMAT_BYTES];
  int vector_width;
  pocl_cuda_printf_length length;
  char conversion;
  int valid;
} pocl_cuda_printf_conversion;

void __cl_va_arg (va_list ap, char data[], int num_words);
int vprintf (const char *, char *);

static int
pocl_cuda_is_digit (char value)
{
  return value >= '0' && value <= '9';
}

static int
pocl_cuda_is_flag (char value)
{
  return value == '-' || value == '+' || value == ' ' || value == '#'
         || value == '0';
}

static int
pocl_cuda_is_conversion (char value)
{
  switch (value)
    {
    case 'd':
    case 'i':
    case 'o':
    case 'u':
    case 'x':
    case 'X':
    case 'f':
    case 'F':
    case 'e':
    case 'E':
    case 'g':
    case 'G':
    case 'a':
    case 'A':
    case 'c':
    case 's':
    case 'p':
      return 1;
    default:
      return 0;
    }
}

static int
pocl_cuda_is_vector_width (int width)
{
  return width == 2 || width == 3 || width == 4 || width == 8
         || width == POCL_CUDA_PRINTF_MAX_VECTOR_WIDTH;
}

static pocl_cuda_printf_conversion
pocl_cuda_parse_conversion (pocl_cuda_format_ptr format)
{
  pocl_cuda_printf_conversion result = { 0 };
  int output_index = 1;
  result.scalar_format[0] = '%';
  result.vector_width = 1;
  result.length = POCL_CUDA_PRINTF_LENGTH_DEFAULT;

  while (pocl_cuda_is_flag (*format) || pocl_cuda_is_digit (*format)
         || *format == '.')
    {
      if (output_index >= POCL_CUDA_PRINTF_FORMAT_BYTES - 1)
        return result;
      result.scalar_format[output_index++] = *format++;
    }

  if (*format == 'v')
    {
      int width = 0;
      ++format;
      while (pocl_cuda_is_digit (*format))
        width = width * 10 + (*format++ - '0');
      if (!pocl_cuda_is_vector_width (width))
        return result;
      result.vector_width = width;
    }

  if (*format == 'h' && *(format + 1) == 'h')
    {
      result.length = POCL_CUDA_PRINTF_LENGTH_HH;
      format += 2;
    }
  else if (*format == 'h' && *(format + 1) == 'l')
    {
      result.length = POCL_CUDA_PRINTF_LENGTH_HL;
      format += 2;
    }
  else if (*format == 'h')
    {
      result.length = POCL_CUDA_PRINTF_LENGTH_H;
      ++format;
    }
  else if (*format == 'l')
    {
      result.length = POCL_CUDA_PRINTF_LENGTH_L;
      result.scalar_format[output_index++] = *format++;
    }

  if (!pocl_cuda_is_conversion (*format)
      || output_index >= POCL_CUDA_PRINTF_FORMAT_BYTES - 1)
    return result;

  result.conversion = *format;
  result.scalar_format[output_index++] = *format++;
  result.scalar_format[output_index] = 0;
  result.next = format;
  result.valid = 1;
  return result;
}

static int
pocl_cuda_is_float_conversion (char conversion)
{
  return conversion == 'f' || conversion == 'F' || conversion == 'e'
         || conversion == 'E' || conversion == 'g' || conversion == 'G'
         || conversion == 'a' || conversion == 'A';
}

static int
pocl_cuda_is_integer_conversion (char conversion)
{
  return conversion == 'd' || conversion == 'i' || conversion == 'o'
         || conversion == 'u' || conversion == 'x' || conversion == 'X';
}

static int
pocl_cuda_emit_scalar_integer (const pocl_cuda_printf_conversion *conversion,
                               const char *raw)
{
  char scalar[8] __attribute__ ((aligned (8)));
  int value = *((const int *)raw);
  int is_signed
      = conversion->conversion == 'd' || conversion->conversion == 'i';
  if (conversion->length == POCL_CUDA_PRINTF_LENGTH_HH)
    {
      if (is_signed)
        *((int *)scalar) = (char)value;
      else
        *((uint *)scalar) = (uchar)value;
    }
  else
    {
      if (is_signed)
        *((int *)scalar) = (short)value;
      else
        *((uint *)scalar) = (ushort)value;
    }
  return vprintf (conversion->scalar_format, scalar);
}

static int
pocl_cuda_emit_scalar (const pocl_cuda_printf_conversion *conversion,
                       const char *raw)
{
  if (pocl_cuda_is_float_conversion (conversion->conversion)
      && conversion->length == POCL_CUDA_PRINTF_LENGTH_H)
    {
      vprintf ("<unsupported half format>", (char *)raw);
      return -1;
    }
  if (pocl_cuda_is_integer_conversion (conversion->conversion)
      && (conversion->length == POCL_CUDA_PRINTF_LENGTH_HH
          || conversion->length == POCL_CUDA_PRINTF_LENGTH_H))
    return pocl_cuda_emit_scalar_integer (conversion, raw);
  return vprintf (conversion->scalar_format, (char *)raw);
}

static int
pocl_cuda_emit_vector_float (const pocl_cuda_printf_conversion *conversion,
                             const char *raw, int element)
{
  char scalar[8] __attribute__ ((aligned (8)));
  double value;
  if (conversion->length == POCL_CUDA_PRINTF_LENGTH_H)
#ifdef cl_khr_fp16
    value = (double)((const half *)raw)[element];
#else
    {
      vprintf ("<unsupported half format>", (char *)raw);
      return -1;
    }
#endif
  else if (conversion->length == POCL_CUDA_PRINTF_LENGTH_L)
    value = ((const double *)raw)[element];
  else
    value = (double)((const float *)raw)[element];
  *((double *)scalar) = value;
  return vprintf (conversion->scalar_format, scalar);
}

static ulong
pocl_cuda_read_vector_unsigned (const char *raw, int element,
                                pocl_cuda_printf_length length)
{
  if (length == POCL_CUDA_PRINTF_LENGTH_HH)
    return ((const uchar *)raw)[element];
  if (length == POCL_CUDA_PRINTF_LENGTH_H)
    return ((const ushort *)raw)[element];
  if (length == POCL_CUDA_PRINTF_LENGTH_L)
    return ((const ulong *)raw)[element];
  return ((const uint *)raw)[element];
}

static long
pocl_cuda_read_vector_signed (const char *raw, int element,
                              pocl_cuda_printf_length length)
{
  if (length == POCL_CUDA_PRINTF_LENGTH_HH)
    return ((const char *)raw)[element];
  if (length == POCL_CUDA_PRINTF_LENGTH_H)
    return ((const short *)raw)[element];
  if (length == POCL_CUDA_PRINTF_LENGTH_L)
    return ((const long *)raw)[element];
  return ((const int *)raw)[element];
}

static int
pocl_cuda_emit_vector_integer (const pocl_cuda_printf_conversion *conversion,
                               const char *raw, int element)
{
  char scalar[8] __attribute__ ((aligned (8)));
  int is_signed
      = conversion->conversion == 'd' || conversion->conversion == 'i';
  if (conversion->length == POCL_CUDA_PRINTF_LENGTH_L)
    {
      if (is_signed)
        *((long *)scalar) = pocl_cuda_read_vector_signed (
            raw, element, conversion->length);
      else
        *((ulong *)scalar) = pocl_cuda_read_vector_unsigned (
            raw, element, conversion->length);
    }
  else if (is_signed)
    *((int *)scalar)
        = (int)pocl_cuda_read_vector_signed (raw, element, conversion->length);
  else
    *((uint *)scalar) = (uint)pocl_cuda_read_vector_unsigned (
        raw, element, conversion->length);
  return vprintf (conversion->scalar_format, scalar);
}

static int
pocl_cuda_emit_vector (const pocl_cuda_printf_conversion *conversion,
                       const char *raw)
{
  for (int element = 0; element < conversion->vector_width; ++element)
    {
      if (element != 0)
        {
          int comma = ',';
          if (vprintf ("%c", (char *)&comma) < 0)
            return -1;
        }
      int result = pocl_cuda_is_float_conversion (conversion->conversion)
                       ? pocl_cuda_emit_vector_float (conversion, raw, element)
                       : pocl_cuda_emit_vector_integer (conversion, raw,
                                                        element);
      if (result < 0)
        return -1;
    }
  return 0;
}

int
printf (pocl_cuda_format_ptr restrict format, ...)
{
  char arg_data[POCL_CUDA_PRINTF_ARG_BYTES] __attribute__ ((aligned (16)));
  va_list ap;
  va_start (ap, format);

  while (*format)
    {
      if (*format != '%')
        {
          int literal = *format++;
          if (vprintf ("%c", (char *)&literal) < 0)
            return -1;
          continue;
        }
      if (*(format + 1) == '%')
        {
          int literal = '%';
          format += 2;
          if (vprintf ("%c", (char *)&literal) < 0)
            return -1;
          continue;
        }

      pocl_cuda_printf_conversion conversion
          = pocl_cuda_parse_conversion (format + 1);
      if (!conversion.valid)
        {
          vprintf ("<format error>", arg_data);
          return -1;
        }

      __cl_va_arg (ap, arg_data, POCL_CUDA_PRINTF_ARG_BYTES / sizeof (int));
      int result = conversion.vector_width == 1
                       ? pocl_cuda_emit_scalar (&conversion, arg_data)
                       : pocl_cuda_emit_vector (&conversion, arg_data);
      if (result < 0)
        return -1;
      format = conversion.next;
    }

  va_end (ap);
  return 0;
}
