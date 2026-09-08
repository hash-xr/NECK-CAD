import os
from datetime import datetime
from core.schema import AnalysisSession, ImageStatus
from traceability.logger import AuditLogger

class ClinicalReportGenerator:
    """
    Generates structured clinical summaries and audit documentation 
    for Fine Needle Aspiration (FNA) cytology analysis.
    """
    def __init__(self, logger: AuditLogger):
        self.logger = logger

    def format_text_report(self, session: AnalysisSession) -> str:
        """Constructs a formatted, human-readable clinical report in British English."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        res = session.aggregated_result
        
        valid_images = [img for img in session.images if img.status == ImageStatus.VALID]
        rejected_images = [img for img in session.images if img.status != ImageStatus.VALID]

        report_lines = [
            "=" * 70,
            "                   NECK-CAD CLINICAL ANALYSIS REPORT                  ",
            "=" * 70,
            f"Session ID:         {session.session_id}",
            f"Report Generated:   {timestamp}",
            f"Total Slide FOVs:   {len(session.images)} ({len(valid_images)} Valid, {len(rejected_images)} Rejected)",
            "-" * 70,
            "1. DIAGNOSTIC SYNTHESIS",
            "-" * 70,
        ]

        if res:
            report_lines.extend([
                f"Primary Categorisation: {res.primary_category}",
                f"Milan System Category:  {res.milan_category}",
                f"Specific Entity:        {res.specific_diagnosis}",
                f"Confidence Scores:      Milan: {res.confidence_scores.get('milan_confidence', 0.0):.2%}, "
                f"Entity: {res.confidence_scores.get('entity_confidence', 0.0):.2%}",
            ])

            if res.differential_diagnosis:
                report_lines.append("\nDifferential Diagnoses:")
                for rank, (entity, prob) in enumerate(res.differential_diagnosis, start=1):
                    report_lines.append(f"  {rank}. {entity:<35} ({prob:.2%})")

            report_lines.extend([
                "\n" + "-" * 70,
                "2. SAFETY & UNCERTAINTY AUDIT",
                "-" * 70,
                f"Prediction Uncertainty Flag: {'HIGH (Manual Review Required)' if res.is_uncertain else 'Normal'}",
                f"Out-of-Distribution (OOD):  {'DETECTED (Unusual Morphological Features)' if res.is_ood else 'Passed'}",
            ])
        else:
            report_lines.append("NO DIAGNOSTIC RESULT GENERATED (Insufficient valid slide data).")

        report_lines.extend([
            "\n" + "-" * 70,
            "3. FIELD-OF-VIEW (FOV) QUALITY BREAKDOWN",
            "-" * 70,
        ])

        for img in session.images:
            status_str = "VALID" if img.status == ImageStatus.VALID else f"REJECTED ({img.rejection_reason})"
            score_str = f"Sharpness: {img.quality_score:.2f}" if img.quality_score > 0 else "N/A"
            report_lines.append(f" - [{img.image_id}] {img.filename:<25} Status: {status_str} | {score_str}")

        report_lines.extend([
            "\n" + "=" * 70,
            "DISCLAIMER: NECK-CAD is a computer-assisted decision support system.",
            "All findings must be validated by a qualified consultant pathologist.",
            "=" * 70,
        ])

        return "\n".join(report_lines)

    def save_report(self, session: AnalysisSession, output_dir: str = "reports") -> str:
        """Saves the clinical summary text file to disk."""
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f"report_{session.session_id}.txt")
        
        try:
            report_content = self.format_text_report(session)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(report_content)
            self.logger.log(f"Clinical report saved successfully to {file_path}.")
            return file_path
        except Exception as e:
            self.logger.log(f"Failed to generate clinical report: {str(e)}", level="ERROR")
            raise e