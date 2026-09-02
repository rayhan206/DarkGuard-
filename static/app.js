const tabs = document.querySelectorAll('.tab');
const urlForm = document.querySelector('#url-form');
const imageForm = document.querySelector('#image-form');
const statusBox = document.querySelector('#status');
const results = document.querySelector('#results');

tabs.forEach((tab) => tab.addEventListener('click', () => {
  tabs.forEach((item) => item.classList.toggle('active', item === tab));
  urlForm.classList.toggle('hidden', tab.dataset.mode !== 'url');
  imageForm.classList.toggle('hidden', tab.dataset.mode !== 'image');
  clearStatus();
}));

urlForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  await requestAnalysis('/api/analyse/url', {
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({url: document.querySelector('#url').value})
  });
});

imageForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  const data = new FormData();
  data.append('image', document.querySelector('#image').files[0]);
  await requestAnalysis('/api/analyse/screenshot', {body: data});
});

async function requestAnalysis(endpoint, options) {
  setStatus('Analysing… this can take a few seconds.');
  results.classList.add('hidden');
  try {
    const response = await fetch(endpoint, {method: 'POST', ...options});
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Analysis failed.');
    renderResults(data);
    setStatus('Analysis complete.');
  } catch (error) {
    setStatus(error.message, true);
  }
}

function renderResults(data) {
  results.classList.remove('hidden');
  document.querySelector('#engine').textContent = data.active_engine;
  document.querySelector('#summary').textContent = data.summary;
  document.querySelector('#original').src = data.original_image;
  document.querySelector('#annotated').src = data.annotated_image;
  document.querySelector('#limitation').textContent = data.limitations.join(' ');
  const cards = document.querySelector('#detections');
  cards.replaceChildren();
  if (!data.detections.length) {
    cards.innerHTML = '<article class="card"><h3>No prominent cues found</h3><p>This screening result is not a guarantee that the interface is free of dark patterns.</p></article>';
    return;
  }
  data.detections.forEach((detection) => {
    const card = document.createElement('article');
    card.className = 'card';
    const evidence = detection.evidence.map((item) => `<li>${escapeText(item)}</li>`).join('');
    card.innerHTML = `<h3>${escapeText(detection.label.replaceAll('_', ' '))}</h3>
      <span class="confidence">${Math.round(detection.confidence * 100)}% confidence</span>
      <p>“${escapeText(detection.text || 'visual element')}”</p>
      <ul>${evidence}</ul><p><strong>Clearer alternative:</strong> ${escapeText(detection.suggestion)}</p>`;
    cards.append(card);
  });
}

function escapeText(value) {
  const node = document.createElement('span');
  node.textContent = value;
  return node.innerHTML;
}

function setStatus(message, isError = false) {
  statusBox.textContent = message;
  statusBox.classList.toggle('error', isError);
}

function clearStatus() { setStatus(''); }
