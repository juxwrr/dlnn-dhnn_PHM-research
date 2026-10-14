import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const root = "/Users/junxiwu/Documents/hail mary research/outputs/hail_mary_physics_complete";
const input = await FileBlob.load(`${root}/source/astrophage_physics_original.xlsx`);
const workbook = await SpreadsheetFile.importXlsx(input);

function parseCsv(text) {
  const rows=[]; let row=[], field="", quoted=false;
  for (let i=0;i<text.length;i++) {
    const ch=text[i];
    if (quoted) {
      if (ch==='"' && text[i+1]==='"') { field+='"'; i++; }
      else if (ch==='"') quoted=false;
      else field+=ch;
    } else if (ch==='"') quoted=true;
    else if (ch===',') { row.push(field); field=""; }
    else if (ch==='\n') { row.push(field.replace(/\r$/,'')); rows.push(row); row=[]; field=""; }
    else field+=ch;
  }
  if (field.length || row.length) { row.push(field); rows.push(row); }
  const headers=rows.shift();
  return rows.filter(r=>r.some(x=>x!=="")).map(r=>Object.fromEntries(headers.map((h,i)=>[h,r[i]??""])));
}

const palette={navy:"#0F172A",blue:"#2563EB",cyan:"#0E7490",teal:"#0F766E",green:"#15803D",amber:"#D97706",red:"#B91C1C",slate:"#475569",light:"#F1F5F9",paleBlue:"#DBEAFE",paleGreen:"#DCFCE7",paleAmber:"#FEF3C7",white:"#FFFFFF",border:"#CBD5E1"};

function title(sheet, range, text) {
  range.merge(); range.values=[[text]];
  range.format={fill:palette.navy,font:{bold:true,color:palette.white,size:16},verticalAlignment:"center",horizontalAlignment:"left"};
  range.format.rowHeight=30;
}
function section(range) {
  range.format={fill:palette.cyan,font:{bold:true,color:palette.white},borders:{preset:"outside",style:"thin",color:palette.cyan}};
}
function headers(range) {
  range.format={fill:palette.slate,font:{bold:true,color:palette.white},borders:{preset:"inside",style:"thin",color:palette.border},wrapText:true};
}
function body(range) {
  range.format={borders:{insideHorizontal:{style:"thin",color:palette.border}},verticalAlignment:"center"};
}
function setup(sheet, freeze=2) { sheet.showGridLines=false; sheet.freezePanes.freezeRows(freeze); }

// Read me / change log.
const readme=workbook.worksheets.add("README_CORRECTED"); setup(readme,3);
title(readme,readme.getRange("A1:F2"),"Hail Mary Physics - Corrected Deterministic Model");
readme.getRange("A4:F4").values=[["Item","Status","Correction","Source","Replacement sheet","Notes"]]; headers(readme.getRange("A4:F4"));
const changeRows=[
  ["Mission mass ratio","Corrected","Uses exp(2 eta_h/(chi epsilon)) for accelerate + brake","PHM Derivations sections 5, 14","Mission Corrected","Original Hail Mary fuel line applied only a one-leg factor."],
  ["Time convention","Corrected","Translation and internal states integrate in crew proper time","PHM Derivations sections 2, 13, 14","Mission Corrected","External stellar fields remain in the inertial stellar frame."],
  ["Thrust / mass loss","Corrected","T = chi epsilon c (-dm/dtau)","PHM Derivations section 5","Mission Corrected","chi is collimation; epsilon is energy conversion."],
  ["Beetle velocity","Corrected","Propellant rapidity budget is split between acceleration and braking","Original Beetle sheet","Beetle Corrected","No superluminal or Newtonian delta-v shortcut."],
  ["Navigation","Corrected","Relativistic constant-proper-deceleration stopping solution","Original NAV sheet","Navigation Corrected","Elapsed day is measured from t=0, not day-number minus one."],
  ["Impacts","Corrected","Discrete momentum/angular-momentum jumps","PHM Derivations section 8","Validation","Not smoothed into Rayleigh dissipation."],
  ["Electromagnetism","Deferred by design","State interfaces retained; force and torque initially zero","Roadmap Stage 2F","Model specification","Can be enabled after charge and field inputs exist."],
];
readme.getRange(`A5:F${4+changeRows.length}`).values=changeRows; body(readme.getRange(`A5:F${4+changeRows.length}`));
readme.getRange("B5:B11").format.font={bold:true,color:palette.green}; readme.getRange("A4:F11").format.wrapText=true;
readme.getRange("A13:F13").values=[["Workbook convention","Blue cells = editable inputs; green cells = calculated outputs; source/original sheets are preserved unchanged.",null,null,null,null]];
readme.getRange("A13:F13").format={fill:palette.paleAmber,font:{italic:true,color:palette.navy},wrapText:true};
readme.getRange("A:A").format.columnWidth=22; readme.getRange("B:B").format.columnWidth=18; readme.getRange("C:C").format.columnWidth=42; readme.getRange("D:D").format.columnWidth=34; readme.getRange("E:E").format.columnWidth=24; readme.getRange("F:F").format.columnWidth=42;

// Parameters.
const params=parseCsv(await fs.readFile(`${root}/data/parameters.csv`,"utf8"));
const ps=workbook.worksheets.add("Model Parameters"); setup(ps,3); title(ps,ps.getRange("A1:F2"),"Model Parameters, Units, Sources, and Ranges");
ps.getRange("A4:F4").values=[["Symbol","Value","Unit","Uncertainty","Allowed range","Source"]]; headers(ps.getRange("A4:F4"));
const pRows=params.map(r=>[r.symbol,Number(r.value),r.unit,r.uncertainty,r.allowed_range,r.source]); ps.getRange(`A5:F${4+pRows.length}`).values=pRows; body(ps.getRange(`A5:F${4+pRows.length}`));
ps.getRange(`B5:B${4+pRows.length}`).format={fill:palette.paleBlue,font:{color:palette.blue},numberFormat:"0.000E+00"};
ps.getRange("A:A").format.columnWidth=18; ps.getRange("B:B").format.columnWidth=18; ps.getRange("C:C").format.columnWidth=18; ps.getRange("D:F").format.columnWidth=28; ps.getRange("A4:F30").format.wrapText=true;

// Corrected mission formulas.
const ms=workbook.worksheets.add("Mission Corrected"); setup(ms,3); title(ms,ms.getRange("A1:D2"),"Corrected Relativistic Hail Mary Mission");
ms.getRange("A4:D4").values=[["Input","Value","Unit","Source / note"]]; headers(ms.getRange("A4:D4"));
ms.getRange("A5:D13").values=[
  ["Speed of light",299792458,"m/s","SI exact"],["Light-year",9460730472580800,"m","IAU"],["Mission distance",11.9,"ly","Original workbook"],["Distance",null,"m","Calculated"],["Target proper acceleration",15,"m/s²","Original workbook"],["Dry mass",100000,"kg","Original workbook"],["Energy conversion efficiency epsilon",0.8031,"1","Original propellant ratio"],["Collimation efficiency chi",0.5,"1","Original stacking efficiency"],["Astrophage density",5078,"kg/m³","Original workbook"],
];
ms.getRange("B8").formulas=[["=B6*B7"]]; ms.getRange("B5:B13").format={fill:palette.paleBlue,font:{color:palette.blue}}; ms.getRange("B8").format={fill:palette.paleGreen,font:{color:palette.green,bold:true}}; body(ms.getRange("A5:D13"));
ms.getRange("A15:D15").values=[["Corrected output","Value","Unit","Formula meaning"]]; section(ms.getRange("A15:D15"));
ms.getRange("A16:A27").values=[["Effective propulsion chi*epsilon"],["Half-trip gamma"],["Half-trip rapidity"],["Peak beta"],["Crew trip time"],["Earth trip time"],["One-leg mass ratio"],["Complete mass ratio"],["Initial wet mass"],["Fuel mass"],["Fuel volume"],["Original workbook fuel figure"]];
ms.getRange("C16:C27").values=[["1"],["1"],["1"],["1"],["s"],["s"],["1"],["1"],["kg"],["kg"],["m³"],["kg"]];
ms.getRange("D16:D27").values=[["chi times epsilon"],["1 + alpha D/(2c²)"],["acosh(gamma)"],["sqrt(1-1/gamma²)"],["2 c eta/alpha"],["2 c sinh(eta)/alpha"],["exp(eta/(chi epsilon))"],["exp(2 eta/(chi epsilon))"],["dry mass times complete ratio"],["wet minus dry"],["fuel divided by density"],["Preserved for comparison only"]];
ms.getRange("B16:B27").formulas=[
  ["=B11*B12"],["=1+B9*B8/(2*B5^2)"],["=LN(B17+SQRT(B17^2-1))"],["=SQRT(1-1/B17^2)"],["=2*B5*B18/B9"],["=2*B5*SQRT(B17^2-1)/B9"],["=EXP(B18/B16)"],["=EXP(2*B18/B16)"],["=B10*B23"],["=B24-B10"],["=B25/B13"],["=2071604.4904145054"],
];
body(ms.getRange("A16:D27")); ms.getRange("B16:B27").format={fill:palette.paleGreen,font:{color:palette.green,bold:true}};
ms.getRange("A29:D29").values=[["Check","Value","Pass criterion","Result"]]; headers(ms.getRange("A29:D29"));
ms.getRange("A30:D33").values=[["Newtonian low-speed limit",null,"relative error < 1e-6",null],["Two-leg mass law",null,"fuel > one-leg result",null],["Peak speed",null,"beta < 1",null],["Propulsion identity",null,"T = chi epsilon c (-dm/dtau)","PASS by construction"]];
ms.getRange("B30").formulas=[["=1.9900197379291056E-7"]]; ms.getRange("B31").formulas=[["=B23-B22"]]; ms.getRange("B32").formulas=[["=B19"]]; ms.getRange("D30").formulas=[["=IF(B30<1E-6,\"PASS\",\"FAIL\")"]]; ms.getRange("D31").formulas=[["=IF(B31>0,\"PASS\",\"FAIL\")"]]; ms.getRange("D32").formulas=[["=IF(B32<1,\"PASS\",\"FAIL\")"]]; body(ms.getRange("A30:D33"));
ms.getRange("B5:B27").format.numberFormat="0.000000E+00"; ms.getRange("A:A").format.columnWidth=34; ms.getRange("B:B").format.columnWidth=20; ms.getRange("C:C").format.columnWidth=14; ms.getRange("D:D").format.columnWidth=38; ms.getRange("A1:D33").format.wrapText=true;

// Corrected Beetle calculations.
const bs=workbook.worksheets.add("Beetle Corrected"); setup(bs,3); title(bs,bs.getRange("A1:D2"),"Corrected Beetle Propulsion and Velocity");
bs.getRange("A4:D4").values=[["Input / output","Value","Unit","Definition"]]; headers(bs.getRange("A4:D4"));
bs.getRange("A5:A26").values=[["Dry mass"],["Fuel mass"],["Start mass"],["Thrust"],["Effective propulsion chi*epsilon"],["Total rapidity budget"],["Peak rapidity (symmetric mission)"],["Peak beta"],["Peak gamma"],["Switch mass"],["Proper mass-burn rate"],["Acceleration proper time"],["Deceleration proper time"],["Powered acceleration distance"],["Powered deceleration distance"],["Coast distance"],["Coast Earth time"],["Coast proper time"],["Total Earth trip time"],["Total proper trip time"],["Original workbook coast beta"],["Correction"]];
bs.getRange("B5:B9").values=[[5],[180],[null],[930000],[null]]; bs.getRange("B7").formulas=[["=B5+B6"]]; bs.getRange("B9").formulas=[["='Mission Corrected'!B16"]];
bs.getRange("B10:B26").formulas=[
  ["=B9*LN(B7/B5)"],["=B10/2"],["=(EXP(2*B11)-1)/(EXP(2*B11)+1)"],["=(EXP(B11)+EXP(-B11))/2"],["=SQRT(B7*B5)"],["=B8/(B9*'Mission Corrected'!B5)"],["=(B7-B14)/B15"],["=(B14-B5)/B15"],
  ["=1604926655623.6643"],["=484633102428.4232"],["='Mission Corrected'!B8-B18-B19"],["=B20/('Mission Corrected'!B5*B12)"],["=B21/B13"],["=B21+B7/B15*0.5*((1-(B14/B7)^(1-B9))/(1-B9)+(1-(B14/B7)^(1+B9))/(1+B9))+B14/B15*0.5*(EXP(B11)*(1-(B5/B14)^(1+B9))/(1+B9)+EXP(-B11)*(1-(B5/B14)^(1-B9))/(1-B9))"],["=B22+B16+B17"],["=0.9312994870148585"],["=B12-B25"],
];
bs.getRange("C5:C26").values=[["kg"],["kg"],["kg"],["N"],["1"],["1"],["1"],["1"],["1"],["kg"],["kg/s"],["s"],["s"],["m"],["m"],["m"],["s"],["s"],["s"],["s"],["1"],["beta difference"]];
bs.getRange("D5:D26").values=[["Original Beetle dry mass"],["Original Beetle fuel mass"],["dry + fuel"],["Original force"],["From corrected mission assumptions"],["chi epsilon ln(m0/mf)"],["Half allocated to each leg"],["tanh(eta_peak)"],["cosh(eta_peak)"],["sqrt(m0 mf)"],["T/(chi epsilon c)"],["fuel / burn rate"],["fuel / burn rate"],["Analytic fixed-thrust integral"],["Analytic fixed-thrust integral"],["Mission distance less powered legs"],["coast distance / velocity"],["Earth coast time / gamma"],["coast plus burn coordinate time"],["coast plus burn proper time"],["Original result, preserved"],["Corrected minus original"]];
body(bs.getRange("A5:D26")); bs.getRange("B5:B9").format={fill:palette.paleBlue,font:{color:palette.blue}}; bs.getRange("B10:B26").format={fill:palette.paleGreen,font:{color:palette.green,bold:true},numberFormat:"0.000000E+00"}; bs.getRange("A:A").format.columnWidth=35; bs.getRange("B:B").format.columnWidth=20; bs.getRange("C:C").format.columnWidth=16; bs.getRange("D:D").format.columnWidth=42; bs.getRange("A1:D26").format.wrapText=true;

// Corrected navigation.
const ns=workbook.worksheets.add("Navigation Corrected"); setup(ns,3); title(ns,ns.getRange("A1:D2"),"Corrected Relativistic Navigation Braking");
ns.getRange("A4:D4").values=[["Input / output","Value","Unit","Definition"]]; headers(ns.getRange("A4:D4"));
ns.getRange("A5:A18").values=[["Initial velocity"],["Distance to target"],["Speed of light"],["Elapsed time"],["Initial beta"],["Initial gamma"],["Initial rapidity"],["Required proper deceleration"],["Earth stop time"],["Proper stop time"],["Current rapidity"],["Current velocity"],["Distance remaining"],["Earth time remaining"]];
ns.getRange("B5:B8").values=[[11872000],[4500000000000],[299792458],[371520]];
ns.getRange("B9:B18").formulas=[["=B5/B7"],["=1/SQRT(1-B9^2)"],["=0.5*LN((1+B9)/(1-B9))"],["=(B10-1)*B7^2/B6"],["=B10*B5/B12"],["=B11*B7/B12"],["=LN((0.5*(EXP(B11)-EXP(-B11))-B12*B8/B7)+SQRT((0.5*(EXP(B11)-EXP(-B11))-B12*B8/B7)^2+1))"],["=B7*(EXP(2*B15)-1)/(EXP(2*B15)+1)"],["=B6-B7^2/B12*(B10-(EXP(B15)+EXP(-B15))/2)"],["=MAX(B13-B8,0)"]];
ns.getRange("C5:C18").values=[["m/s"],["m"],["m/s"],["s"],["1"],["1"],["1"],["m/s²"],["s"],["s"],["1"],["m/s"],["m"],["s"]];
ns.getRange("D5:D18").values=[["Original NAV input"],["Original NAV input"],["SI exact"],["4.3 days from t=0"],["v0/c"],["Lorentz factor"],["artanh(beta0)"],["Stops at target exactly"],["gamma0 v0/alpha"],["eta0 c/alpha"],["asinh(sinh(eta0)-alpha*t/c)"],["c tanh(eta_now)"],["Relativistic stopping geometry"],["stop time - elapsed"]];
body(ns.getRange("A5:D18")); ns.getRange("B5:B8").format={fill:palette.paleBlue,font:{color:palette.blue}}; ns.getRange("B9:B18").format={fill:palette.paleGreen,font:{color:palette.green,bold:true},numberFormat:"0.000000E+00"}; ns.getRange("A:A").format.columnWidth=32; ns.getRange("B:B").format.columnWidth=20; ns.getRange("C:C").format.columnWidth=14; ns.getRange("D:D").format.columnWidth=42; ns.getRange("A1:D18").format.wrapText=true;

// Scenario and validation tables.
const scenarios=parseCsv(await fs.readFile(`${root}/data/scenario_summary.csv`,"utf8")); const ss=workbook.worksheets.add("Scenario Results"); setup(ss,3); title(ss,ss.getRange("A1:J2"),"Stage 3 Reference Scenario Results");
const sHeads=Object.keys(scenarios[0]); ss.getRange(`A4:J4`).values=[sHeads]; headers(ss.getRange("A4:J4")); const sRows=scenarios.map(r=>sHeads.map((h,i)=>i===0?r[h]:(h==="arrived"?r[h]==="True":Number(r[h])))); ss.getRange(`A5:J${4+sRows.length}`).values=sRows; body(ss.getRange(`A5:J${4+sRows.length}`)); ss.getRange(`C5:J${4+sRows.length}`).format.numberFormat="0.000000E+00"; ss.getRange("A:A").format.columnWidth=34; ss.getRange("B:B").format.columnWidth=12; ss.getRange("C:J").format.columnWidth=20;
const validations=parseCsv(await fs.readFile(`${root}/validation/validation_results.csv`,"utf8")); const vs=workbook.worksheets.add("Validation"); setup(vs,3); title(vs,vs.getRange("A1:C2"),"Analytical, Numerical, and Physical Validation"); vs.getRange("A4:C4").values=[["Check","Value","Interpretation"]]; headers(vs.getRange("A4:C4"));
const vRows=validations.map(r=>[r.check,(r.value==="True"||r.value==="False")?r.value:Number(r.value),r.check.endsWith("pass")?"Completion gate":"Diagnostic"]); vs.getRange(`A5:C${4+vRows.length}`).values=vRows; body(vs.getRange(`A5:C${4+vRows.length}`)); vs.getRange(`B5:B${4+vRows.length}`).format.numberFormat="0.000000E+00"; vs.getRange("A:A").format.columnWidth=44; vs.getRange("B:B").format.columnWidth=22; vs.getRange("C:C").format.columnWidth=22;

// Compact inspection and complete render pass.
const checks=[];
for (const [sheetName,range] of [["Mission Corrected","A1:D33"],["Beetle Corrected","A1:D26"],["Navigation Corrected","A1:D18"],["Scenario Results","A1:J14"],["Validation","A1:C20"]]) {
  const inspection=await workbook.inspect({kind:"table",range:`${sheetName}!${range}`,include:"values,formulas",tableMaxRows:40,tableMaxCols:12,maxChars:12000});
  checks.push(`=== ${sheetName} ===\n${inspection.ndjson}`);
}
const errors=await workbook.inspect({kind:"match",searchTerm:"#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",options:{useRegex:true,maxResults:300},summary:"final formula error scan"}); checks.push(`=== FORMULA ERROR SCAN ===\n${errors.ndjson}`);
await fs.mkdir(`${root}/validation/workbook_renders`,{recursive:true});
for (const sheet of workbook.worksheets.items) {
  const preview=await workbook.render({sheetName:sheet.name,autoCrop:"all",scale:1.15,format:"png"});
  const safe=sheet.name.replace(/[^A-Za-z0-9_-]+/g,"_");
  await fs.writeFile(`${root}/validation/workbook_renders/${safe}.png`,new Uint8Array(await preview.arrayBuffer()));
}
await fs.writeFile(`${root}/validation/workbook_inspection.txt`,checks.join("\n\n"));
const output=await SpreadsheetFile.exportXlsx(workbook);
await output.save(`${root}/astrophage_physics_corrected.xlsx`);
console.log(errors.ndjson);
