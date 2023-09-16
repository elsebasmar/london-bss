.DEFAULT_GOAL := default
#################### PACKAGE ACTIONS ###################

run_api:
	uvicorn londonbss.api.fast:app --reload
