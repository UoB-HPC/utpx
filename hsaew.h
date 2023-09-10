#pragma once

#include <cstddef>
#include <cstdint>

extern "C" {

typedef enum {
  HSA_STATUS_SUCCESS = 0x0,
} hsa_status_t;

typedef struct hsa_executable_symbol_s {
  uint64_t handle;
} hsa_executable_symbol_t;

typedef struct hsa_executable_s {
  uint64_t handle;
} hsa_executable_t;

typedef struct hsa_code_object_reader_s {
  uint64_t handle;
} hsa_code_object_reader_t;

typedef struct hsa_agent_s {
  uint64_t handle;
} hsa_agent_t;

typedef struct hsa_signal_s {
  uint64_t handle;
} hsa_signal_t;

typedef int64_t hsa_signal_value_t;

typedef enum {
  HSA_EXECUTABLE_SYMBOL_INFO_TYPE = 0,
  HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH = 1,
  HSA_EXECUTABLE_SYMBOL_INFO_NAME = 2,
  HSA_EXECUTABLE_SYMBOL_INFO_MODULE_NAME_LENGTH = 3,
  HSA_EXECUTABLE_SYMBOL_INFO_MODULE_NAME = 4,
  HSA_EXECUTABLE_SYMBOL_INFO_AGENT = 20,
  HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ADDRESS = 21,
  HSA_EXECUTABLE_SYMBOL_INFO_LINKAGE = 5,
  HSA_EXECUTABLE_SYMBOL_INFO_IS_DEFINITION = 17,
  HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ALLOCATION = 6,
  HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_SEGMENT = 7,
  HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ALIGNMENT = 8,
  HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_SIZE = 9,
  HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_IS_CONST = 10,
  HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT = 22,
  HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE = 11,
  HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT = 12,
  HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE = 13,
  HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE = 14,
  HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_DYNAMIC_CALLSTACK = 15,
  HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_CALL_CONVENTION = 18,
  HSA_EXECUTABLE_SYMBOL_INFO_INDIRECT_FUNCTION_OBJECT = 23,
  HSA_EXECUTABLE_SYMBOL_INFO_INDIRECT_FUNCTION_CALL_CONVENTION = 16
} hsa_executable_symbol_info_t;

typedef enum
{
  HSA_SYSTEM_INFO_VERSION_MAJOR = 0,
  HSA_SYSTEM_INFO_VERSION_MINOR = 1,
  HSA_SYSTEM_INFO_TIMESTAMP = 2,
  HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY = 3,
  HSA_SYSTEM_INFO_SIGNAL_MAX_WAIT = 4,
  HSA_SYSTEM_INFO_ENDIANNESS = 5,
  HSA_SYSTEM_INFO_MACHINE_MODEL = 6,
  HSA_SYSTEM_INFO_EXTENSIONS = 7,
  HSA_AMD_SYSTEM_INFO_BUILD_VERSION = 0x200,
  HSA_AMD_SYSTEM_INFO_SVM_SUPPORTED = 0x201,
  HSA_AMD_SYSTEM_INFO_SVM_ACCESSIBLE_BY_DEFAULT = 0x202
} hsa_system_info_t;
typedef hsa_status_t (*_hsa_system_get_info)(hsa_system_info_t  , void * );

typedef hsa_status_t (*_hsa_code_object_reader_create_from_memory)(const void *, size_t, hsa_code_object_reader_t *);

typedef hsa_status_t (*_hsa_executable_symbol_get_info)(hsa_executable_symbol_t, hsa_executable_symbol_info_t, void *);

typedef hsa_status_t (*_hsa_executable_get_symbol_by_name)(hsa_executable_t executable, const char *symbol_name, const hsa_agent_t *agent,
                                                           hsa_executable_symbol_t *symbol);

typedef hsa_signal_value_t (*_hsa_signal_load_relaxed)(hsa_signal_t);
}