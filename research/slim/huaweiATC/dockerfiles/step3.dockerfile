FROM 0234439a534a
ENV PATH=/usr/local/python3.7.5/bin:/usr/local/Ascend/ascend-toolkit/20.1.rc1/atc/ccec_compiler/bin:/usr/local/Ascend/ascend-toolkit/20.1.rc1/atc/bin:$PATH
ENV PYTHONPATH=/usr/local/Ascend/ascend-toolkit/20.1.rc1/atc/python/site-packages/te:/usr/local/Ascend/ascend-toolkit/20.1.rc1/atc/python/site-packages/topi:$PYTHONPATH
ENV LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/20.1.rc1/atc/lib64:$LD_LIBRARY_PATH
ENV ASCEND_OPP_PATH=/usr/local/Ascend/ascend-toolkit/20.1.rc1/opp

