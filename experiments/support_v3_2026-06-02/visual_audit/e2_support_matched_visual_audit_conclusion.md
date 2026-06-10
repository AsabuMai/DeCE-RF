# E2.4 Visual Audit Conclusion

Manual review of the seed10/seed11 support-matched grids confirms that post-hoc mask blending improves preservation metrics but frequently misses the requested edit or introduces visible paste boundaries.
DeCE-RF is stronger on target correctness and overall quality. Fixed DeCE displacement is treated as a separate component ablation, so E2.4 stays focused on whether localization alone explains the gain.

Audit CSV: experiments/support_v3_2026-06-02/visual_audit/e2_support_matched_visual_audit_filled.csv
Summary CSV: experiments/support_v3_2026-06-02/visual_audit/e2_support_matched_visual_audit_summary.csv
Combined table: experiments/support_v3_2026-06-02/e2_support_matched_contextual_table_with_audit.md
