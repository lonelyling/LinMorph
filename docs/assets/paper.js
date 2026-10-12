'use strict';
const teaser=document.querySelector('#showcase-video');
if(teaser){
  if(matchMedia('(prefers-reduced-motion: reduce)').matches)teaser.autoplay=false;
  document.addEventListener('visibilitychange',()=>{if(document.hidden)teaser.pause();});
}
