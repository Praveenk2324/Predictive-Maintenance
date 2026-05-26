import pandas as pd
import random
import os
from datetime import datetime, timedelta

# --- Configurations ---
OUTPUT_DIR = "data/raw"
NUM_LOGS = 1500

# Mapping components to real ATA chapters and our specific CMAPSS sensors
COMPONENTS = [
    {"name": "High-Pressure Turbine (HPT)", "ata": "72-50", "sensors": ["s11", "s12", "s13", "s14"]},
    {"name": "Low-Pressure Turbine (LPT)", "ata": "72-40", "sensors": ["s15", "s17"]},
    {"name": "High-Pressure Compressor (HPC)", "ata": "72-30", "sensors": ["s4", "s7", "s8", "s9"]},
    {"name": "Fan Module / Bypass", "ata": "72-20", "sensors": ["s2", "s3"]},
    {"name": "Combustor Section", "ata": "72-40", "sensors": ["s11", "s12"]},
    {"name": "Engine Control (FADEC)", "ata": "73-20", "sensors": ["s20", "s21"]},
    {"name": "Oil System", "ata": "79-00", "sensors": ["s18", "s19"]},
    {"name": "Fuel System", "ata": "73-10", "sensors": ["s6", "s20"]},
    {"name": "Inlet / Nacelle", "ata": "71-10", "sensors": ["s1", "s2"]},
    {"name": "Exhaust / Nozzle", "ata": "78-30", "sensors": ["s8", "s13"]},
    {"name": "Low-Pressure Compressor (LPC / Booster)", "ata": "72-20", "sensors": ["s3", "s4"]},
    {"name": "Accessory Gearbox (AGB)", "ata": "72-60", "sensors": ["s18", "s19"]},
    {"name": "Turbine Rear Frame (TRF)", "ata": "72-50", "sensors": ["s14", "s15"]},
    {"name": "Variable Stator Vane (VSV) System", "ata": "72-30", "sensors": ["s7", "s9"]},
    {"name": "Bleed Air / Anti-Ice System", "ata": "36-10", "sensors": ["s10", "s16"]},
    {"name": "Thrust Reverser", "ata": "78-30", "sensors": ["s20", "s21"]},
    {"name": "Starter / Ignition System", "ata": "74-10", "sensors": ["s20", "s21"]},
    {"name": "Engine Mounts / Pylon", "ata": "71-20", "sensors": ["s1"]},
]

# --- Realistic Findings (40+) ---
FINDINGS = [
    # HPT / LPT
    "Borescope inspection revealed Thermal Barrier Coating (TBC) spallation on HPT stage-1 blades.",
    "HPT stage-2 nozzle guide vanes exhibit oxidation and hot-section erosion beyond serviceable limits.",
    "LPT blade tip rub marks identified; rotor-to-shroud clearance measured at 0.003 in, exceeding the 0.002 in maximum.",
    "Trailing edge cracking noted on HPT stage-1 blades during 800-cycle hard-time borescope.",
    "Foreign Object Damage (FOD) dent found on HPT stage-2 blade leading edge; depth measured at 0.015 in.",
    "Inter-turbine temperature (ITT) exceedance recorded — peak of 987 °C against a redline of 960 °C during climb.",
    "LPT stage-3 shroud segment shows circumferential crack; confirmed serviceable under AMM 72-40-05 blend limits.",
    # HPC
    "Trend monitoring flagged sustained deviation in bleed air temperature (+18 °C over 30-cycle baseline).",
    "HPC stage-7 blade tip erosion reducing surge margin below EHM threshold.",
    "Rub strip contact observed on HPC stage-5 rotor; abradable coating partially displaced.",
    "HPC casing distortion causing variable stator vane (VSV) schedule anomaly at high power settings.",
    "EGT margin deterioration of 22 °C noted over last 200 cycles; HPC wash scheduled.",
    "Micro-cracking observed on root attachment of HPC stage-2 blades; confirmed by fluorescent penetrant inspection (FPI).",
    # Fan / LPC
    "Fan blade leading edge nick measuring 0.08 in width found at 60% span during walk-around.",
    "Impact damage to fan spinner cone; crack propagating from rivet hole confirmed by dye-penetrant check.",
    "Fan case acoustic liner panel delamination over a 6 × 4 inch area; water ingress suspected.",
    "LPC booster blade erosion causing inlet distortion and increased N1 vibration.",
    "Fan blade dovetail fretting wear observed; fits within blend limits per AMM 72-20-03.",
    # Combustor
    "Debris found in magnetic chip detector (MCD); metallurgical analysis confirms #3 bearing wear particles.",
    "Excessive vibration recorded on the #2 bearing structural frame — 3.2 IPS pk-pk against limit of 2.5 IPS.",
    "Acoustic anomaly reported by flight crew during takeoff roll; traced to combustor liner burnthrough.",
    "Heavy coking and carbon buildup observed on fuel nozzle atomizers — 4 of 24 nozzles below flow spec.",
    "Combustor liner TBC spallation exposing parent metal in hot-spot region; liner at life limit.",
    "Hot streak on turbine inlet temperature (TIT) sensor trace indicates circumferential non-uniformity in combustion.",
    # Oil System
    "Oil consumption trend exceeding 0.5 qt/hr; on-wing investigation reveals #4 bearing carbon seal degradation.",
    "Oil filter bypass indicator (OFBI) popped in-flight; filter element found heavily loaded with metallic debris.",
    "Oil pressure fluctuation between 40–55 PSI during cruise; suspected air ingestion at oil tank vent.",
    "Chip light illuminated during engine ground run; chip confirmed as single ferrous sliver <1 mm.",
    "Oil cooler matrix found cracked at inlet header, causing low oil pressure indication.",
    "AGB oil filler cap found improperly seated; oil loss estimated at 2.3 qt over 4-hour flight.",
    # Fuel System
    "Pressure drop exceeding 15 PSI across the primary fuel manifold; filter element blocked.",
    "Fuel flowmeter reading 4% low vs. calibrated master; suspected sensor contamination.",
    "Fuel nozzle #12 flow rate at 87% of nominal; nozzle pulled and replaced.",
    "P&D valve failed to close during engine shutdown; fuel dribble into combustor confirmed.",
    "High-pressure fuel pump discharge pressure 200 PSI below spec at max power; pump worn.",
    "Water-in-fuel (WIF) sensor triggered during pre-flight; 0.5 cc of water found in low-point drain.",
    # FADEC / Sensors
    "FADEC logged 14 EICAS maintenance messages relating to T4.5 thermocouple channel mismatch.",
    "N2 speed sensor signal intermittent above FL350; EMI source identified on P-lead harness.",
    "EGT thermocouple harness chafing against fuel line; two conductors show bare wire.",
    "FADEC software exception code 0x4A31 — fuel control loop saturation at altitude relight attempt.",
    "Sensor recalibration required: PS3 static pressure probe reading 2% high versus pitot-static reference.",
    "T2 inlet temperature sensor iced over during descent, causing erroneous FADEC fuel schedule trim.",
    # Bleed Air / Anti-Ice
    "Customer bleed air duct found cracked at flex-joint flange; hot-air leak confirmed by thermal camera.",
    "5th-stage bleed check valve stuck open; high-pressure bleed air back-flowing into 9th-stage duct.",
    "Anti-ice valve butterfly disc fractured; nacelle anti-ice inoperative — deferred per MEL 30-11.",
    "Bleed duct insulation blanket missing from 18-inch section; risk of overheat on adjacent hydraulic line.",
    # VSV System
    "Actuator linkage found binding during manual operational check; jam nut backed off causing misalignment.",
    "VSV schedule error of +3.5° at N2 = 87%; FADEC trim corrected after actuator feedback cable re-rigged.",
    "VSV unison ring wear pin fractured; complete ring replaced per AMM 72-30-12.",
    # Thrust Reverser
    "Thrust reverser sleeve failed to stow fully; blocker door hinge pin corroded and seized.",
    "In-flight deployment warning light illuminated; reverser cowl latch safety switch found open-circuit.",
    "Thrust reverser hydraulic actuator internal leak rate measured at 18 cc/min against limit of 5 cc/min.",
    # Starter / Ignition
    "Air turbine starter (ATS) drive shaft shear coupling failed during attempted engine start.",
    "Igniter plug A shows excessive electrode erosion; spark gap measured at 0.18 in (limit 0.12 in).",
    "Engine failed to light off within hung-start time limit; ignition exciter output checked at 1.8 J vs. 3.0 J spec.",
    # Mounts / Pylon
    "Engine mount forward link bushings worn beyond serviceable limits; free-play measured at 0.025 in.",
    "Pylon secondary structure skin crack found during heavy maintenance visit; crack length 3.2 in.",
    "Fan cowl door latch pin fractured; door opened in flight causing structural damage to leading edge.",
    # Exhaust / Nozzle
    "EGT margin trending -30 °C over 500 cycles; borescope confirms nozzle area growth due to tip rub.",
    "Core exhaust nozzle divergent flap hinge fractured; flap found kinematically jammed.",
    "Turbine Rear Frame strut cracks detected by eddy-current inspection; 3 of 10 struts affected.",
    # Miscellaneous
    "Excessive vibration N1 (1.8 IPS pk) traced to single fan blade mass imbalance; field balance performed.",
    "Seal degradation resulting in higher-than-expected secondary airflow bypass; labyrinth knife-edges worn.",
    "Recurring high EGT on engine #2 linked to compressor wash interval — on-wing wash completed.",
    "Engine refused to accelerate above 80% N1 during takeoff; bleed air valve incorrectly rigged.",
    "Oil-in-fuel indication post-maintenance; #3 bearing sump seal misassembled during last shop visit.",
    "ACARS transmitted unsolicited engine exceedance event; review confirmed sensor spike, not true exceedance.",
    "Hydraulic actuator seal failure causing hydraulic fluid contamination in nacelle zone 3.",
    "Post-bird-strike inspection revealed fan blade FOD on blades 4, 7, and 11; all within blend limits.",
]

# --- Realistic Actions (40+) ---
ACTIONS = [
    "Component removed and replaced with serviceable unit (PN: {pn}). Ground engine run and ops check normal.",
    "Deferred per MEL {ata}-01. Performance monitoring required every 50 flight cycles; trend reviewed at each check.",
    "Performed deep chemical wash and re-torqued casing bolts to 450 in-lbs per AMM {ata}-00-03.",
    "Removed and routed to off-site MRO shop for full overhaul (work order WO-{pn}). Installed serviceable spare.",
    "Sensor recalibrated against master gauge (serial #{pn}). Ground engine run at max continuous power — all params normal.",
    "Blended out compressor blade damage per AMM 72-30-05; within all dimensional limits. FPI re-check clear.",
    "Applied high-temp RTV sealant (MIL-A-46146) and safety-wired mounting brackets per engineering order.",
    "Adjusted fuel-flow metering within FADEC software limits (ref SB {ata}-73-12). Cleared for revenue service.",
    "Performed on-wing borescope per AMM task 72-00-00-290-801. No further deterioration; component returned to service.",
    "Replaced O-ring seals (PN: {pn}) and torqued coupling nuts to 75 ft-lbs. Pressure-checked to 150 PSI — no leaks.",
    "Performed MCD chip analysis per Engineering Order EO-{pn}; single isolated particle cleared by metallurgy. Ops check passed.",
    "Carried out compressor on-wing wash per CFMI SB 72-0XXX. EGT margin recovered by 18 °C. Returned to service.",
    "Replaced fuel nozzle set (24 nozzles, PN: {pn}). Engine ground run at idle and max continuous — EGT spread within 15 °C.",
    "VSV rigging adjusted per AMM 72-30-12 to restore schedule within ±1°. Functional test satisfactory.",
    "Thermal camera survey performed per AMM 36-10-00; bleed air leak confirmed and duct replaced (PN: {pn}).",
    "Igniter plug replaced (PN: {pn}). Ignition system verified per AMM 74-10-00 — light-off within 10 seconds.",
    "Fan blade blended per AMM 72-20-03 within all chord and depth limits. Post-blend vibration survey: 0.4 IPS — acceptable.",
    "Engine mount bushings replaced per AMM 71-20-01 (PN: {pn}). Alignment checked; free-play within 0.008 in limit.",
    "Thrust reverser actuator replaced (PN: {pn}). Full stow/deploy cycle tests × 5 — all within time limits.",
    "Anti-ice valve replaced (PN: {pn}) and system leak-checked at 45 PSI. Functional test confirmed valve cycling.",
    "Starter replaced (PN: {pn}); motoring check confirmed N2 rise rate meets AMM minimum. Start cycle tested × 3.",
    "Oil cooler replaced (PN: {pn}). System pressure-tested at 90 PSI for 10 minutes — zero leak. Oil serviced and engine run.",
    "EGT harness replaced (PN: {pn}). All thermocouple channels checked against reference — max split 12 °C. FADEC reset.",
    "Combustor liner replaced with serviceable unit (PN: {pn}). Borescope post-installation confirms correct seating.",
    "High-pressure fuel pump replaced (PN: {pn}). Discharge pressure confirmed at 2,950 PSI at max continuous — within spec.",
    "Fan cowl latch replaced (PN: {pn}). All 6 latches tested for positive engagement and safety catch. Ops check complete.",
    "Field fan balance performed (4 balance weights added per AMM 72-20-09). N1 vibration reduced from 1.8 to 0.3 IPS.",
    "LPT shroud segment blended within limits per AMM 72-40-07. Borescope re-check confirms crack arrest. Returned to service.",
    "FADEC software updated to version 5.1.4 per SB {ata}-73-056. Ground test cycle confirms correct engine response.",
    "Bleed check valve replaced (PN: {pn}). Bleed system functional test at ground idle and max continuous — normal.",
    "Turbine Rear Frame strut repair per DER-approved repair scheme RS-{pn}. NDT re-check clear. Returned to service.",
    "Engine sent to test cell for full performance restoration workscope. Post-test EGT margin: +42 °C. Returned to fleet.",
    "Water-in-fuel system drained and flushed per AMM 73-10-07. Filter element replaced (PN: {pn}). WIF sensor re-tested OK.",
    "P&D valve replaced (PN: {pn}). Engine shutdown test × 3 confirms full closure within 1.5 s. Leak check clear.",
    "Oil seal replaced (PN: {pn}). Engine oil consumption check over 10-hour period: 0.15 qt/hr — within limit of 0.4 qt/hr.",
    "Hydraulic actuator seal kit replaced (PN: {pn}). Internal leakage re-tested at 0.8 cc/min — well within 5 cc/min limit.",
    "Nose cowl anti-ice duct insulation blanket replaced per AMM 30-21-01. Adjacent structure inspected — no heat damage found.",
    "N2 sensor and wiring harness replaced (PN: {pn}). Signal continuity confirmed on all channels. Engine run at cruise power — stable.",
    "Pylon skin crack repaired per SRM Chapter 53 approved doubler repair. NDT re-check: no crack extension. Returned to service.",
    "Engine inlet FOD inspection completed per post-bird-strike check sheet. All blades within blend limits. Returned to service.",
    "Fan acoustic liner panel repaired with structural film adhesive per AMM 71-10-04 (PN: {pn}). Tap test confirms no further delamination.",
    "T2 sensor de-iced and anti-ice inlet heater power supply verified. FADEC fault cleared. Altitude relight test successful.",
    "Unison ring replaced (PN: {pn}) per AMM 72-30-12. VSV schedule verified across N2 range — max deviation 0.8°.",
]

# --- Log templates (varied natural language) ---
LOG_TEMPLATES = [
    "ATA {ata} | Engine {eng}: {finding} {action}",
    "Pilot Report — Engine {eng} ({comp}): {finding} Corrective Action: {action}",
    "Auto-Alert on Sensor {sensor} (Engine {eng}). {finding} {action}",
    "Routine Inspection of {comp} on Engine {eng}: {finding} Action Taken: {action}",
    "Line Maintenance Write-Up #{log_id} | {comp} (ATA {ata}): {finding} Resolution: {action}",
    "Shop Finding — Engine S/N {eng} | {comp}: {finding} Disposition: {action}",
    "EHM System Flag — Sensor {sensor} | Engine {eng} | ATA {ata}: {finding} {action}",
    "Non-Routine Card NRC-{log_id} opened for Engine {eng} {comp}. Finding: {finding} Action: {action}",
    "Maintenance Debrief Engine {eng} ({comp}, ATA {ata}): Crew reported anomaly. Finding: {finding} {action}",
    "ACARS Engine Alert — Engine {eng} Sensor {sensor}: {finding} Ground team response: {action}",
    "Heavy Check C-Visit Finding | Engine {eng} | {comp}: {finding} Engineering disposition: {action}",
    "Engine {eng} AOG Recovery — {comp} (ATA {ata}): {finding} Expedited action: {action}",
    "Trend Analysis Report — Engine {eng} {comp}: {finding} Maintenance response: {action}",
    "Post-Flight Inspection Engine {eng} (ATA {ata}): {finding} {action}",
    "Borescope Work Order | Engine {eng} | {comp}: {finding} Outcome: {action}",
]

def generate_logs(num_logs):
    logs = []
    base_date = datetime.now() - timedelta(days=365)

    for i in range(1, num_logs + 1):
        engine_id = random.randint(1, 100)
        comp_data = random.choice(COMPONENTS)

        component = comp_data["name"]
        ata = comp_data["ata"]
        sensor = random.choice(comp_data["sensors"])
        finding = random.choice(FINDINGS)

        pn = f"{random.randint(100, 999)}-{random.choice(['A', 'B', 'C', 'X', 'Z'])}{random.randint(10, 99)}"
        action = random.choice(ACTIONS).format(pn=pn, ata=ata)

        template = random.choice(LOG_TEMPLATES)
        log_num = 1000 + i
        log_text = template.format(
            ata=ata,
            eng=engine_id,
            comp=component,
            sensor=sensor.upper(),
            finding=finding,
            action=action,
            log_id=log_num,
        )

        log_date = base_date + timedelta(days=random.randint(0, 365), hours=random.randint(0, 23))

        logs.append({
            "log_id": f"LOG-{log_num}",
            "date": log_date.strftime("%Y-%m-%d %H:%M"),
            "engine_id": engine_id,
            "ata_chapter": ata,
            "component": component,
            "text": log_text,
        })

    return pd.DataFrame(logs)


def main():
    print(f"Generating {NUM_LOGS} realistic synthetic maintenance logs...")
    df = generate_logs(NUM_LOGS)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "maintenance_logs.csv")
    df.to_csv(output_path, index=False)

    print(f"Saved to {output_path}")
    print(f"\nDataset shape: {df.shape}")
    print(f"Unique components: {df['component'].nunique()}")
    print(f"Unique ATA chapters: {df['ata_chapter'].nunique()}")
    print("\n--- Sample Logs ---")
    for i in range(5):
        print(f"\n{i+1}. [{df.iloc[i]['log_id']}] {df.iloc[i]['text']}")


if __name__ == "__main__":
    main()