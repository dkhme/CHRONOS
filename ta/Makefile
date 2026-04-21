CFG_TEE_TA_LOG_LEVEL ?= 2

# The UUID for the CHRONOS TA
BINARY=8a1b2c3d-4e5f-6a7b-8c9d-0e1f2a3b4c5d

-include $(TA_DEV_KIT_DIR)/mk/ta_dev_kit.mk

ifeq ($(wildcard $(TA_DEV_KIT_DIR)/mk/ta_dev_kit.mk), )
clean:
	@echo 'Note: $$(TA_DEV_KIT_DIR)/mk/ta_dev_kit.mk not found, cannot clean TA.'
	@echo 'Note: TA_DEV_KIT_DIR=$(TA_DEV_KIT_DIR)'
endif
