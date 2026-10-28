const $=id=>document.getElementById(id); let D, view="backtest", chart;
const NAMES={seasonal_naive:"Seasonal naive (baseline)",gbm_lags:"Gradient boosting + lags"};
const mape=(a,p)=>a.reduce((s,x,i)=>s+Math.abs((x-p[i])/x),0)/a.length*100;
const rmse=(a,p)=>Math.sqrt(a.reduce((s,x,i)=>s+(x-p[i])**2,0)/a.length);
const f=x=>x.toLocaleString(undefined,{maximumFractionDigits:2});
function render(){
  const s=D.series[$("series").value], h=+$("h").value; $("hval").textContent=h; document.querySelectorAll(".hh").forEach(e=>e.textContent=h);
  $("desc").textContent=s.description;
  const c=s.cutoff_index, actual=s.values.slice(c+1,c+1+h);
  const res=Object.keys(NAMES).map(k=>({k,mape:mape(actual,s.holdout[k].slice(0,h)),rmse:rmse(actual,s.holdout[k].slice(0,h))}));
  const best=res.reduce((a,b)=>a.mape<b.mape?a:b), base=res[0], gbm=res[1];
  const lift=(1-gbm.mape/base.mape)*100;
  $("kpis").innerHTML=[["Baseline MAPE",f(base.mape)+"%"],["Model MAPE",f(gbm.mape)+"%"],["Error reduction vs baseline",(lift>=0?"−":"+")+f(Math.abs(lift))+"%"],["Winner",NAMES[best.k]]]
    .map(([l,v])=>`<div class="kpi"><div class="l">${l}</div><div class="v" style="font-size:${v.length>14?15:22}px">${v}</div></div>`).join("");
  const row=(r,isBest)=>`<tr><td>${NAMES[r.k]}</td><td class="num">${f(r.mape)}%</td><td class="num">${f(r.rmse)}</td><td>${isBest?'<span class="win">best</span>':""}</td></tr>`;
  const hdr='<tr><th>Model</th><th class="num">MAPE</th><th class="num">RMSE</th><th></th></tr>';
  $("m1").innerHTML=hdr+res.map(r=>row(r,r.k===best.k)).join("");
  const rb=Object.keys(NAMES).map(k=>({k,...s.rolling_backtest[k]})); const b2=rb.reduce((a,b)=>a.mape<b.mape?a:b);
  $("m2").innerHTML=hdr+rb.map(r=>row(r,r.k===b2.k)).join("");
  let labels, ds;
  if(view==="backtest"){
    const start=Math.max(0,c-35); labels=s.dates.slice(start,c+1+h);
    const pad=c+1-start, nul=Array(pad).fill(null);
    ds=[{label:"Actual",data:s.values.slice(start,c+1+h),borderColor:"#e7ecf7",pointRadius:0},
        {label:NAMES.seasonal_naive,data:nul.concat(s.holdout.seasonal_naive.slice(0,h)),borderColor:"#93a0bd",borderDash:[5,5],pointRadius:2},
        {label:NAMES.gbm_lags,data:nul.concat(s.holdout.gbm_lags.slice(0,h)),borderColor:"#5b8cff",pointRadius:2}];
    $("chartTitle").textContent=`Backtest: models trained up to ${s.dates[c]}, forecasting ${h} months, compared with what actually happened`;
  } else {
    const n=s.values.length, start=n-36; labels=s.dates.slice(start).concat(s.future_dates.slice(0,h));
    const nul=Array(36).fill(null);
    ds=[{label:"History",data:s.values.slice(start),borderColor:"#e7ecf7",pointRadius:0},
        {label:NAMES.seasonal_naive,data:nul.concat(s.future.seasonal_naive.slice(0,h)),borderColor:"#93a0bd",borderDash:[5,5],pointRadius:2},
        {label:NAMES.gbm_lags,data:nul.concat(s.future.gbm_lags.slice(0,h)),borderColor:"#5b8cff",pointRadius:2}];
    $("chartTitle").textContent=`Future forecast: next ${h} months from ${s.dates[n-1]}`;
  }
  if(window.Chart){Chart.defaults.color="#93a0bd";Chart.defaults.borderColor="#24304d"; if(chart) chart.destroy();
    chart=new Chart($("chart"),{type:"line",data:{labels,datasets:ds},options:{interaction:{mode:"index",intersect:false},spanGaps:false}});}
  $("curl").textContent=`docker build -t forecast . && docker run -p 8000:8000 forecast\n\ncurl -X POST localhost:8000/forecast -H 'Content-Type: application/json' \\\n  -d '{"series": "${$("series").value}", "horizon": ${h}}'`;
}
fetch("data.json").then(r=>r.json()).then(d=>{D=d; $("ver").textContent=d.model_version;
  $("series").innerHTML=Object.keys(d.series).map(k=>`<option value="${k}">${k.replace(/_/g," ")}</option>`).join("");
  $("series").value="retail_sales"; $("series").onchange=render; $("h").oninput=render;
  document.querySelectorAll(".seg button").forEach(b=>b.onclick=()=>{view=b.dataset.v;document.querySelectorAll(".seg button").forEach(x=>x.classList.toggle("on",x===b));render();});
  render();});
