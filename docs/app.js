const $=id=>document.getElementById(id); let D, view="backtest", chart;
const NAMES={seasonal_naive:"Seasonal naive (baseline)",gbm_lags:"Gradient boosting + lags",ets_simple:"ETS simple (SES)",ets_holt:"ETS Holt (trend)",ets_hw_add:"Holt-Winters additive"};
const COLORS={seasonal_naive:"#93a0bd",gbm_lags:"#5b8cff",ets_simple:"#c38bff",ets_holt:"#ffb35b",ets_hw_add:"#2ecc8f"};
const mape=(a,p)=>a.reduce((s,x,i)=>s+Math.abs((x-p[i])/x),0)/a.length*100;
const smape=(a,p)=>a.reduce((s,x,i)=>s+2*Math.abs(x-p[i])/(Math.abs(x)+Math.abs(p[i])||1),0)/a.length*100;
const rmse=(a,p)=>Math.sqrt(a.reduce((s,x,i)=>s+(x-p[i])**2,0)/a.length);
const f=x=>x.toLocaleString(undefined,{maximumFractionDigits:2});
const pct=x=>Math.round(x*100)+"%";
const hexA=(h,a)=>`rgba(${parseInt(h.slice(1,3),16)},${parseInt(h.slice(3,5),16)},${parseInt(h.slice(5,7),16)},${a})`;
// Prediction-interval band for the selected model: lower line (invisible) + upper line filled down to it.
const band=(b,nul,h,lvl,col)=>b?[{label:`${pct(lvl)} interval (lower)`,data:nul.concat(b.lower.slice(0,h)),borderColor:"rgba(0,0,0,0)",pointRadius:0,fill:false},
  {label:`${pct(lvl)} prediction interval (selected model)`,data:nul.concat(b.upper.slice(0,h)),borderColor:"rgba(0,0,0,0)",backgroundColor:hexA(col,.16),pointRadius:0,fill:"-1"}]:[];
function lines(src,nul,h,sel){
  // Selected model drawn bold with its band; baseline dashed; other candidates hidden (click the legend to show).
  return D.models.map(k=>({label:NAMES[k]+(k===sel?" ★ selected":""),data:nul.concat(src[k].slice(0,h)),borderColor:COLORS[k],
    borderWidth:k===sel?3:1.5,borderDash:k==="seasonal_naive"?[5,5]:[],pointRadius:k===sel?2:0,hidden:!(k===sel||k==="seasonal_naive")}));
}
function render(){
  const s=D.series[$("series").value], h=+$("h").value; $("hval").textContent=h; document.querySelectorAll(".hh").forEach(e=>e.textContent=h);
  $("desc").textContent=s.description;
  const sh=s.selection.holdout.winner, sf=s.selection.future.winner, sel=view==="backtest"?sh:sf;
  const c=s.cutoff_index, actual=s.values.slice(c+1,c+1+h);
  const res=D.models.map(k=>({k,mape:mape(actual,s.holdout[k].slice(0,h)),smape:smape(actual,s.holdout[k].slice(0,h)),rmse:rmse(actual,s.holdout[k].slice(0,h))}));
  const best=res.reduce((a,b)=>a.mape<b.mape?a:b), base=res.find(r=>r.k==="seasonal_naive"), pick=res.find(r=>r.k===sh);
  const lift=(1-pick.mape/base.mape)*100, lvl=D.interval_level||0.8, hb=s.holdout_band[sh];
  const kp=[["Auto-selected (train window only)",NAMES[sh]],["Baseline MAPE",f(base.mape)+"%"],["Selected model MAPE",f(pick.mape)+"%"],
    ["Error reduction vs baseline",(lift>=0?"−":"+")+f(Math.abs(lift))+"%"],["Best on these months",NAMES[best.k]]];
  const inside=actual.filter((a,i)=>a>=hb.lower[i]&&a<=hb.upper[i]).length;
  kp.push([`${pct(lvl)} interval coverage (target ${pct(lvl)})`,`${inside}/${actual.length} = ${pct(inside/actual.length)}`]);
  $("kpis").innerHTML=kp
    .map(([l,v])=>`<div class="kpi"><div class="l">${l}</div><div class="v" style="font-size:${v.length>14?15:22}px">${v}</div></div>`).join("");
  const tag=(k,isBest,selK)=>(k===selK?'<span class="tag">selected</span> ':"")+(isBest?'<span class="win">best</span>':"");
  const row=(r,isBest,selK)=>`<tr><td>${NAMES[r.k]}</td><td class="num">${f(r.mape)}%</td><td class="num">${f(r.smape)}%</td><td class="num">${f(r.rmse)}</td><td>${tag(r.k,isBest,selK)}</td></tr>`;
  const hdr='<tr><th>Model</th><th class="num">MAPE</th><th class="num">sMAPE</th><th class="num">RMSE</th><th></th></tr>';
  $("m1").innerHTML=hdr+[...res].sort((a,b)=>a.mape-b.mape).map(r=>row(r,r.k===best.k,sh)).join("");
  const rb=s.selection.future.leaderboard.map(r=>({k:r.model,...r}));
  $("m2").innerHTML=hdr+rb.map((r,i)=>row(r,i===0,sf)).join("");
  let labels, ds;
  if(view==="backtest"){
    const start=Math.max(0,c-35); labels=s.dates.slice(start,c+1+h);
    const pad=c+1-start, nul=Array(pad).fill(null);
    ds=[{label:"Actual",data:s.values.slice(start,c+1+h),borderColor:"#e7ecf7",pointRadius:0},...lines(s.holdout,nul,h,sh),...band(hb,nul,h,lvl,COLORS[sh])];
    $("chartTitle").textContent=`Backtest: models trained up to ${s.dates[c]}, forecasting ${h} months, compared with what actually happened`;
  } else {
    const n=s.values.length, start=n-36; labels=s.dates.slice(start).concat(s.future_dates.slice(0,h));
    const nul=Array(36).fill(null);
    ds=[{label:"History",data:s.values.slice(start),borderColor:"#e7ecf7",pointRadius:0},...lines(s.future,nul,h,sf),...band(s.future_band[sf],nul,h,lvl,COLORS[sf])];
    $("chartTitle").textContent=`Future forecast: next ${h} months from ${s.dates[n-1]} (production pick: ${NAMES[sf]})`;
  }
  $("ver").textContent=s.model_version;
  if(window.Chart){Chart.defaults.color="#93a0bd";Chart.defaults.borderColor="#24304d"; if(chart) chart.destroy();
    chart=new Chart($("chart"),{type:"line",data:{labels,datasets:ds},options:{interaction:{mode:"index",intersect:false},spanGaps:false,
      plugins:{legend:{labels:{filter:i=>!i.text.endsWith("(lower)")}}}}});}
  $("curl").textContent=`docker build -t forecast . && docker run -p 8000:8000 forecast\n\ncurl localhost:8000/models/${$("series").value}      # backtest leaderboard\n\ncurl -X POST localhost:8000/forecast -H 'Content-Type: application/json' \\\n  -d '{"series": "${$("series").value}", "horizon": ${h}, "model": "auto"}'`;
}
fetch("data.json").then(r=>r.json()).then(d=>{D=d;
  $("series").innerHTML=Object.keys(d.series).map(k=>`<option value="${k}">${k.replace(/_/g," ")}</option>`).join("");
  $("series").value="retail_sales"; $("series").onchange=render; $("h").oninput=render;
  document.querySelectorAll(".seg button").forEach(b=>b.onclick=()=>{view=b.dataset.v;document.querySelectorAll(".seg button").forEach(x=>x.classList.toggle("on",x===b));render();});
  render();});
