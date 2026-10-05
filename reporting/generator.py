# reporting/generator.py
import os
from datetime import datetime
from core.schema import DiagnosticSession, ImageStatus
from traceability.logger import AuditLogger

class ClinicalReportGenerator:
    """
    Generates structured clinical summaries and electronic health record (EHR) 
    documentation for Fine Needle Aspiration (FNA) cytology salivary gland cases.
    """
    def __init__(self, logger: AuditLogger):
        self.logger = logger

    def format_text_report(self, session: DiagnosticSession) -> str:
        """Constructs a professional clinical report using a completely flat text builder."""
        timestamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        res = session.aggregated_result
        
        valid_count = len([img for img in session.images if img.status == ImageStatus.VALID])
        rejected_count = len([img for img in session.images if img.status != ImageStatus.VALID])

        # Start with flat layout blocks
        report_lines = [
            "======================================================================",
            "                   NECK-CAD CLINICAL ANALYSIS REPORT                  ",
            "         Salivary Gland Fine Needle Aspiration (FNA) Analytics        ",
            "======================================================================",
            f"Session ID:          {session.session_id}",
            f"Analysis Timestamp:  {timestamp}",
            f"Total Target FOVs:   {len(session.images)} ({valid_count} Valid, {rejected_count} Rejected)",
            "----------------------------------------------------------------------",
            " 1. AGGREGATED DIAGNOSTIC SYNTHESIS",
            "----------------------------------------------------------------------",
        ]

        if res:
            # Extract basic metric strings
            report_lines.append(f" Primary Categorisation:      {res.primary_category}")
            report_lines.append(f" Milan System Category:       {res.milan_category}")
            report_lines.append(f" Predicted Specific Neoplasm:  {res.specific_diagnosis}")
            report_lines.append("\n Mathematical Confidence Signatures:")

            # Safe scalar variable extraction
            p_val = res.confidence_scores.get('primary_confidence', 0.0) * 100.0
            m_val = res.confidence_scores.get('milan_confidence', 0.0) * 100.0
            e_val = res.confidence_scores.get('entity_confidence', 0.0) * 100.0

            report_lines.append(f"  - Primary Classification Head: {p_val:.2f}%")
            report_lines.append(f"  - Milan Risk Stratification:   {m_val:.2f}%")
            report_lines.append(f"  - Specific Entity Classifier:   {e_val:.2f}%")

            # Flat Differential Matrix Loop
            if res.differential_diagnosis:
                report_lines.append("\n Complete Differential Diagnosis Matrix:")
                report_lines.append("   Rank | Category / Entity | Probability")
                report_lines.append("   -------------------------------------------------------------------")
                for rank, (entity, prob) in enumerate(res.differential_diagnosis, start=1):
                    p_pct = prob * 100.0
                    report_lines.append(f"   {rank} | {entity} | {p_pct:.2f}%")

            # Uncertainty variable tracking
            entropy_score = res.confidence_scores.get('shannon_entropy', 0.0)
            energy_score = res.confidence_scores.get('free_energy', 0.0)
            
            if res.is_uncertain:
                u_txt = "HIGH (Manual Review Required)"
            else:
                u_txt = "Normal"
                
            if res.is_ood:
                o_txt = "DETECTED (Unusual Morphological Features)"
            else:
                o_txt = "Passed"

            report_lines.append("\n----------------------------------------------------------------------")
            report_lines.append(" 2. ALGORITHMIC UNCERTAINTY & OOD AUDITING METRICS")
            report_lines.append("----------------------------------------------------------------------")
            report_lines.append(f" Shannon Predictive Entropy:    {entropy_score:.4f}")
            report_lines.append(f" Free Energy Out-of-Dist Score: {energy_score:.4f}")
            report_lines.append(f" Prediction Uncertainty Flag:   {u_txt}")
            report_lines.append(f" Out-of-Distribution (OOD):     {o_txt}")
        else:
            report_lines.append(" CRITICAL STATUS: NO CLINICAL SYNTHESIS GENERATED (Insufficient valid data).")

        report_lines.append("\n----------------------------------------------------------------------")
        report_lines.append(" 3. INDIVIDUAL FIELD-OF-VIEW (FOV) QUALITY TRACKING LOGS")
        report_lines.append("----------------------------------------------------------------------")

        # Flat loop context block with simple appending
        for img in session.images:
            if img.status == ImageStatus.VALID:
                status_txt = "VALID"
            else:
                status_txt = f"REJECTED ({img.rejection_reason})"

            q_score = img.quality_score
            report_lines.append(f" - [{img.image_id}] Filename: {img.filename} | Status: {status_txt} | Sharpness: {q_score:.2f}")

        report_lines.append("\n======================================================================")
        report_lines.append(" NHS REGULATORY COMPLIANCE NOTE:")
        report_lines.append(" NECK-CAD is configured strictly as a computer-assisted diagnosis")
        report_lines.append(" support system. All mathematical inferences and structural feature")
        report_lines.append(" summaries must be independently verified by a consultant pathologist.")
        report_lines.append("======================================================================")

        return "\n".join(report_lines)

    def save_report(self, session: DiagnosticSession, output_dir: str = "reports") -> str:
        """Saves a structured clinical summary text transcript file straight to local disk."""
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