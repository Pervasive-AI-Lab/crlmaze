#!/bin/bash
# simple script for testing the models. We should integrate this in the code
# later.

xvfb-run python src/main.py with cfgs/object/test_object_agent0.json >> object_test_res.txt
xvfb-run python src/main.py with cfgs/object/test_object_agent1.json >> object_test_res.txt
xvfb-run python src/main.py with cfgs/object/test_object_agent2.json >> object_test_res.txt