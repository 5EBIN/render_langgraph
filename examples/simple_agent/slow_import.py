"""Failure: timeout -- simulates heavy module-level work (model/vectorstore
build) that never finishes within the hard timeout."""
import time

time.sleep(120)

graph = None
