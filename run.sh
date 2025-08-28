# # HEALTHY - iSH WIP
# python3 src/test.py pathological 8 --opt \
#     batch_duration=4  \
#     gpi_g_str=0.03    \
#     gpi_g_g_g=0.02    \
#     gpi_g_s_g=0.1     \
#     gpe_g_str=0.01    \
#     gpe_g_g_g=0.2     \
#     gpe_g_s_g=0.03    \
#     stn_g_g_s=4.5     \
#     --plt s v

# # PATHOLOGY SLOW s-networs-network
# python3 src/test.py pathological 8 --opt \
#     batch_duration=4  \
#     gpi_g_str=0.03    \
#     gpi_g_g_g=0.02    \
#     gpi_g_s_g=0.1     \
#     gpe_g_str=0.01    \
#     gpe_g_g_g=0.02    \
#     gpe_g_s_g=0.03    \
#     stn_g_g_s=4.5     \
#     --plt s v

# # PATHOLOGY FAST s-network
# python3 src/test.py pathological 8 --opt \
#     batch_duration=4  \
#     gpi_g_str=0.03    \
#     gpi_g_g_g=0.02    \
#     gpi_g_s_g=0.1     \
#     gpe_g_str=0.01    \
#     gpe_g_g_g=0.2    \
#     gpe_g_s_g=0.03    \
#     stn_g_g_s=1.0     \
#     --plt s v


# ctx
python3 src/test.py pathological 3 --opt \
    batch_duration=3  \
    gpi_g_str=0.03    \
    gpi_g_g_g=0.02    \
    gpi_g_s_g=0.1     \
    gpe_g_str=0.01    \
    gpe_g_g_g=0.2     \
    gpe_g_s_g=0.03    \
    stn_g_g_s=4.5     \
    --plt s v
