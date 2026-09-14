<script setup>
import { computed, onMounted, ref } from 'vue'
import { api } from './lib/api.js'

const tabs = ['Chat', 'Knowledge', 'Search', 'Skills', 'Monitor', 'Eval']
const activeTab = ref('Chat')

// ---------------- Health strip (shared, refreshed on demand) ----------------
const health = ref(null)
const healthErr = ref('')
async function refreshHealth() {
  try {
    health.value = await api.health()
    healthErr.value = ''
  } catch (e) {
    healthErr.value = e.message
  }
}

// ---------------- Chat panel ----------------
const chatUser = ref('web-user')
const chatConv = ref('')
const chatUseRag = ref(true)
const chatInput = ref('')
const chatBusy = ref(false)
const chatErr = ref('')
const chatLog = ref([])

async function sendChat() {
  const message = chatInput.value.trim()
  if (!message || chatBusy.value) return
  chatBusy.value = true
  chatErr.value = ''
  chatLog.value.push({ role: 'user', content: message })
  chatInput.value = ''
  try {
    const resp = await api.chat({
      message,
      userId: chatUser.value || 'web',
      conversationId: chatConv.value || null,
      useRag: chatUseRag.value,
    })
    chatConv.value = resp.conversation_id
    chatLog.value.push({ role: 'assistant', content: resp.answer, meta: resp })
  } catch (e) {
    chatErr.value = e.message
    chatLog.value.push({ role: 'system', content: `error: ${e.message}` })
  } finally {
    chatBusy.value = false
  }
}

function resetConversation() {
  chatConv.value = ''
  chatLog.value = []
  chatErr.value = ''
}

function topN(scores, n = 3) {
  if (!scores) return []
  return Object.entries(scores)
    .sort((a, b) => b[1] - a[1])
    .slice(0, n)
    .map(([k, v]) => ({ k, v: Number(v).toFixed(3) }))
}
function urgencyClass(urgency) {
  if (urgency === 'high') return 'danger'
  if (urgency === 'low') return 'success'
  return 'warn'
}

// ---------------- Knowledge panel ----------------
const kbStats = ref(null)
const kbText = ref('')
const kbSource = ref('')
const kbBusy = ref(false)
const kbErr = ref('')
const kbNotice = ref('')
const fileInput = ref(null)

async function refreshKbStats() {
  try {
    kbStats.value = await api.knowledgeStats()
    kbErr.value = ''
  } catch (e) {
    kbErr.value = e.message
  }
}

async function addKb() {
  if (!kbText.value.trim() || kbBusy.value) return
  kbBusy.value = true
  kbErr.value = ''
  kbNotice.value = ''
  try {
    const resp = await api.knowledgeAdd([
      { text: kbText.value, metadata: kbSource.value ? { source: kbSource.value } : {} },
    ])
    kbNotice.value = `Added ${resp.count} document(s) (id=${resp.ids[0]})`
    kbText.value = ''
    kbSource.value = ''
    await refreshKbStats()
  } catch (e) {
    kbErr.value = e.message
  } finally {
    kbBusy.value = false
  }
}

async function uploadKb(evt) {
  const file = evt.target.files?.[0]
  if (!file) return
  kbBusy.value = true
  kbErr.value = ''
  kbNotice.value = ''
  try {
    const resp = await api.knowledgeUpload(file)
    kbNotice.value = `Uploaded ${resp.source} (${resp.bytes} B, id=${resp.id})`
    await refreshKbStats()
  } catch (e) {
    kbErr.value = e.message
  } finally {
    kbBusy.value = false
    if (fileInput.value) fileInput.value.value = ''
  }
}

// ---------------- Search panel ----------------
const searchQuery = ref('how do refunds work?')
const searchTopK = ref(5)
const searchRewriteN = ref(3)
const searchRerank = ref(true)
const searchBusy = ref(false)
const searchResult = ref(null)
const searchErr = ref('')

async function runSearch() {
  if (!searchQuery.value.trim() || searchBusy.value) return
  searchBusy.value = true
  searchErr.value = ''
  try {
    searchResult.value = await api.search({
      query: searchQuery.value,
      topK: Number(searchTopK.value),
      rewriteN: Number(searchRewriteN.value),
      rerank: searchRerank.value,
    })
  } catch (e) {
    searchErr.value = e.message
  } finally {
    searchBusy.value = false
  }
}

// ---------------- Skills panel ----------------
const skillsList = ref([])
const skillsErr = ref('')
const skillsBusy = ref(false)

async function refreshSkills() {
  skillsBusy.value = true
  skillsErr.value = ''
  try {
    const resp = await api.skills()
    skillsList.value = resp.skills
  } catch (e) {
    skillsErr.value = e.message
  } finally {
    skillsBusy.value = false
  }
}
async function reloadSkills() {
  skillsBusy.value = true
  skillsErr.value = ''
  try {
    const resp = await api.skillsReload()
    skillsList.value = resp.skills
  } catch (e) {
    skillsErr.value = e.message
  } finally {
    skillsBusy.value = false
  }
}

// ---------------- Monitor panel ----------------
const monitor = ref(null)
const monitorErr = ref('')
const monitorBusy = ref(false)

async function refreshMonitor() {
  monitorBusy.value = true
  monitorErr.value = ''
  try {
    const [m, h] = await Promise.all([api.monitor(), api.health()])
    monitor.value = { ...m, health: h }
  } catch (e) {
    monitorErr.value = e.message
  } finally {
    monitorBusy.value = false
  }
}

const monitorAgents = computed(() => {
  // Backend shape: monitor.monitor.components["agent:<name>"] = {samples, success_rate, ...}
  // Penalty lives at monitor.monitor.agent_penalties[name].
  const components = monitor.value?.monitor?.components || {}
  const penalties = monitor.value?.monitor?.agent_penalties || {}
  return Object.entries(components)
    .filter(([key]) => key.startsWith('agent:'))
    .map(([key, stats]) => {
      const name = key.slice('agent:'.length)
      return { name, penalty: penalties[name] ?? 0, ...stats }
    })
})
const monitorTools = computed(() => {
  const counters = monitor.value?.tools?.counters || {}
  const breakers = monitor.value?.tools?.breakers || {}
  return Object.entries(counters).map(([name, c]) => ({
    name,
    breaker: breakers[name] || 'closed',
    ...c,
  }))
})

// ---------------- Eval panel ----------------
const evalResult = ref(null)
const evalBusy = ref(false)
const evalErr = ref('')

async function runEval() {
  evalBusy.value = true
  evalErr.value = ''
  try {
    evalResult.value = await api.evalRun()
  } catch (e) {
    evalErr.value = e.message
  } finally {
    evalBusy.value = false
  }
}

const evalCases = computed(() => evalResult.value?.cases || evalResult.value?.results || [])
const evalSummary = computed(() => {
  if (!evalResult.value) return null
  // Backend shape: {passed, metrics: {...}, baseline, regressions, cases}
  return {
    passed: evalResult.value.passed,
    ...(evalResult.value.metrics || {}),
    regressions: (evalResult.value.regressions || []).length,
  }
})

// ---------------- lifecycle ----------------
onMounted(async () => {
  await refreshHealth()
})

async function switchTab(tab) {
  activeTab.value = tab
  if (tab === 'Knowledge' && !kbStats.value) await refreshKbStats()
  if (tab === 'Skills' && !skillsList.value.length) await refreshSkills()
  if (tab === 'Monitor' && !monitor.value) await refreshMonitor()
}
</script>

<template>
  <div class="container">
    <div class="header">
      <div>
        <h1>MyEchoMind Console</h1>
        <div class="sub">Multi-agent orchestration with RAG, memory, monitoring and evaluation</div>
      </div>
      <div class="row">
        <span v-if="health" class="chip accent">llm={{ health.llm_mode }}</span>
        <span v-if="health" class="chip">emb={{ health.embedding_mode }}</span>
        <span v-if="health" class="chip">kb={{ health.knowledge_base?.count ?? 0 }}</span>
        <span v-if="health" class="chip">skills={{ health.skills }}</span>
        <button class="ghost" @click="refreshHealth">refresh</button>
      </div>
    </div>

    <div v-if="healthErr" class="error">Health check failed: {{ healthErr }}. Is the backend running on :8000?</div>

    <div class="tabs">
      <button
        v-for="t in tabs"
        :key="t"
        :class="{ active: activeTab === t }"
        @click="switchTab(t)"
      >{{ t }}</button>
    </div>

    <!-- ============================================================ Chat -->
    <section v-show="activeTab === 'Chat'">
      <div class="panel">
        <h2>Conversation</h2>
        <div class="row" style="margin-bottom:10px">
          <label class="row" style="gap:6px">user_id
            <input type="text" v-model="chatUser" style="width:160px" />
          </label>
          <label class="row" style="gap:6px">conversation_id
            <input type="text" v-model="chatConv" placeholder="auto" style="width:220px" />
          </label>
          <label class="row" style="gap:6px">
            <input type="checkbox" v-model="chatUseRag" /> use RAG
          </label>
          <button class="ghost" @click="resetConversation">new conversation</button>
        </div>

        <div class="chat-log">
          <div v-for="(m, i) in chatLog" :key="i" :class="['msg', m.role]">
            <div>{{ m.content }}</div>
            <div v-if="m.meta" class="meta">
              <span class="chip accent">intent: {{ m.meta.intent.intent }} / {{ m.meta.intent.group }}</span>
              <span :class="['chip', urgencyClass(m.meta.intent.urgency)]">
                urgency: {{ m.meta.intent.urgency }}
              </span>
              <span class="chip">conf: {{ Number(m.meta.intent.confidence).toFixed(3) }}</span>
              <span class="chip">primary: {{ m.meta.routing.primary_agent }}</span>
              <span v-if="m.meta.routing.supporting_agents.length" class="chip">
                supporting: {{ m.meta.routing.supporting_agents.join(', ') }}
              </span>
              <span v-if="m.meta.routing.escalation_requested" class="chip danger">escalation</span>
              <span class="chip">rag: {{ m.meta.knowledge.enabled ? 'on' : 'off' }}</span>
              <span class="chip">{{ m.meta.latency_ms }} ms</span>
            </div>
            <details v-if="m.meta">
              <summary>diagnostics</summary>
              <pre>routing_reason: {{ m.meta.routing.routing_reason }}
domain_signals: {{ JSON.stringify(m.meta.routing.domain_signals) }}
monitor_penalties: {{ JSON.stringify(m.meta.routing.monitor_penalties) }}
entities: {{ JSON.stringify(m.meta.intent.entities) }}
top intents (fused): {{ JSON.stringify(topN(m.meta.intent.fusion_scores?.fused)) }}
rag docs: {{ (m.meta.knowledge.tool?.data?.docs || []).length }}</pre>
            </details>
          </div>
          <div v-if="!chatLog.length" class="msg system">Send a message to start.</div>
        </div>

        <div class="row" style="margin-top:12px">
          <input
            type="text"
            v-model="chatInput"
            placeholder="Type your message"
            @keydown.enter="sendChat"
            style="flex:1"
          />
          <button @click="sendChat" :disabled="chatBusy">
            {{ chatBusy ? 'sending...' : 'send' }}
          </button>
        </div>
        <div v-if="chatErr" class="error">{{ chatErr }}</div>
      </div>
    </section>

    <!-- ======================================================== Knowledge -->
    <section v-show="activeTab === 'Knowledge'">
      <div class="panel">
        <h2>Knowledge base</h2>
        <div class="row" style="margin-bottom:12px">
          <span v-if="kbStats" class="chip accent">docs: {{ kbStats.count }}</span>
          <span v-if="kbStats" class="chip">collection: {{ kbStats.name }}</span>
          <button class="ghost" @click="refreshKbStats">refresh</button>
        </div>
        <div v-if="kbErr" class="error">{{ kbErr }}</div>
        <div v-if="kbNotice" class="notice">{{ kbNotice }}</div>

        <div class="grid two">
          <div>
            <h3 style="margin:0 0 8px;font-size:13px">Add text</h3>
            <textarea v-model="kbText" placeholder="Paste knowledge content..." rows="6"></textarea>
            <input type="text" v-model="kbSource" placeholder="source (optional)" style="margin-top:8px" />
            <button style="margin-top:8px" @click="addKb" :disabled="kbBusy || !kbText.trim()">add</button>
          </div>
          <div>
            <h3 style="margin:0 0 8px;font-size:13px">Upload file</h3>
            <p style="color:var(--fg-muted);font-size:12px;margin:0 0 8px">
              Accepts .txt, .md, .json up to 10 MB (UTF-8).
            </p>
            <input ref="fileInput" type="file" accept=".txt,.md,.json" @change="uploadKb" />
          </div>
        </div>
      </div>
    </section>

    <!-- =========================================================== Search -->
    <section v-show="activeTab === 'Search'">
      <div class="panel">
        <h2>RAG search</h2>
        <div class="row" style="margin-bottom:10px">
          <input type="text" v-model="searchQuery" style="flex:1" @keydown.enter="runSearch" />
          <label class="row" style="gap:6px">top_k
            <input type="number" v-model.number="searchTopK" min="1" max="20" style="width:70px" />
          </label>
          <label class="row" style="gap:6px">rewrite_n
            <input type="number" v-model.number="searchRewriteN" min="1" max="5" style="width:70px" />
          </label>
          <label class="row" style="gap:6px">
            <input type="checkbox" v-model="searchRerank" /> rerank
          </label>
          <button @click="runSearch" :disabled="searchBusy">
            {{ searchBusy ? 'searching...' : 'search' }}
          </button>
        </div>
        <div v-if="searchErr" class="error">{{ searchErr }}</div>

        <div v-if="searchResult">
          <div class="row" style="margin-bottom:10px">
            <span class="chip" :class="searchResult.ok ? 'success' : 'danger'">
              ok={{ searchResult.ok }}
            </span>
            <span v-if="searchResult.degraded" class="chip warn">degraded</span>
            <span class="chip">cache_hit={{ searchResult.meta?.cache_hit ? 'yes' : 'no' }}</span>
            <span class="chip">elapsed_ms={{ searchResult.meta?.elapsed_ms ?? '-' }}</span>
            <span class="chip">variants={{ searchResult.meta?.variants ?? '-' }}</span>
          </div>

          <div v-if="searchResult.data?.rewrites?.length" style="margin-bottom:12px">
            <div style="color:var(--fg-muted);font-size:12px;margin-bottom:4px">rewrites</div>
            <div class="row">
              <span v-for="(r, i) in searchResult.data.rewrites" :key="i" class="chip">{{ r }}</span>
            </div>
          </div>

          <div>
            <div style="color:var(--fg-muted);font-size:12px;margin-bottom:6px">documents</div>
            <div v-for="d in searchResult.data?.docs || []" :key="d.id" class="doc-card">
              <div class="id">{{ d.id }}</div>
              <div class="body">{{ d.text }}</div>
              <div class="footer">
                <span v-if="d.rerank_score !== undefined">rerank={{ Number(d.rerank_score).toFixed(3) }}</span>
                <span v-if="d.distance !== null && d.distance !== undefined">
                  distance={{ Number(d.distance).toFixed(3) }}
                </span>
                <span v-for="(v, k) in d.metadata || {}" :key="k">{{ k }}={{ v }}</span>
              </div>
            </div>
            <div v-if="!(searchResult.data?.docs || []).length" class="msg system">
              No documents returned.
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- =========================================================== Skills -->
    <section v-show="activeTab === 'Skills'">
      <div class="panel">
        <h2>Skills</h2>
        <div class="row" style="margin-bottom:10px">
          <button class="ghost" @click="refreshSkills" :disabled="skillsBusy">refresh</button>
          <button @click="reloadSkills" :disabled="skillsBusy">reload from disk</button>
          <span class="chip">count: {{ skillsList.length }}</span>
        </div>
        <div v-if="skillsErr" class="error">{{ skillsErr }}</div>
        <table v-if="skillsList.length">
          <thead>
            <tr>
              <th>name</th>
              <th>agents</th>
              <th>keywords</th>
              <th>path</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="s in skillsList" :key="s.name || s.id">
              <td class="mono">{{ s.name || s.id }}</td>
              <td>{{ (s.agents || []).join(', ') || '-' }}</td>
              <td>{{ (s.keywords || []).slice(0, 6).join(', ') }}{{ (s.keywords || []).length > 6 ? '...' : '' }}</td>
              <td class="mono" style="color:var(--fg-muted)">{{ s.source_path || s.path || s.source || '' }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>

    <!-- ========================================================== Monitor -->
    <section v-show="activeTab === 'Monitor'">
      <div class="panel">
        <h2>Monitor</h2>
        <div class="row" style="margin-bottom:10px">
          <button class="ghost" @click="refreshMonitor" :disabled="monitorBusy">refresh</button>
        </div>
        <div v-if="monitorErr" class="error">{{ monitorErr }}</div>

        <div v-if="monitor" class="grid two">
          <div>
            <h3 style="margin:0 0 6px;font-size:13px">Agents</h3>
            <table v-if="monitorAgents.length">
              <thead>
                <tr>
                  <th>agent</th><th>samples</th><th>success</th><th>avg ms</th><th>p95 ms</th><th>penalty</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="a in monitorAgents" :key="a.name">
                  <td class="mono">{{ a.name }}</td>
                  <td>{{ a.samples ?? 0 }}</td>
                  <td>{{ a.success_rate !== undefined ? (a.success_rate * 100).toFixed(1) + '%' : '-' }}</td>
                  <td>{{ a.avg_latency_ms?.toFixed(0) ?? '-' }}</td>
                  <td>{{ a.p95_latency_ms?.toFixed(0) ?? '-' }}</td>
                  <td>{{ a.penalty?.toFixed(2) ?? '0.00' }}</td>
                </tr>
              </tbody>
            </table>
            <div v-else class="msg system">No agent activity yet.</div>
          </div>

          <div>
            <h3 style="margin:0 0 6px;font-size:13px">Tools</h3>
            <table>
              <thead>
                <tr>
                  <th>tool</th><th>calls</th><th>hits</th><th>fail</th><th>rejected</th><th>cache</th><th>breaker</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="t in monitorTools" :key="t.name">
                  <td class="mono">{{ t.name }}</td>
                  <td>{{ t.calls }}</td>
                  <td>{{ t.hits }}</td>
                  <td>{{ t.failures }}</td>
                  <td>{{ t.rejected }}</td>
                  <td>{{ t.cache_hits }}</td>
                  <td>
                    <span :class="['chip', t.breaker === 'open' ? 'danger' : 'success']">
                      {{ t.breaker }}
                    </span>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        <details v-if="monitor" style="margin-top:12px">
          <summary>raw monitor payload</summary>
          <pre>{{ JSON.stringify(monitor, null, 2) }}</pre>
        </details>
      </div>
    </section>

    <!-- ============================================================= Eval -->
    <section v-show="activeTab === 'Eval'">
      <div class="panel">
        <h2>End-to-end evaluation</h2>
        <div class="row" style="margin-bottom:10px">
          <button @click="runEval" :disabled="evalBusy">
            {{ evalBusy ? 'running...' : 'run default cases' }}
          </button>
        </div>
        <div v-if="evalErr" class="error">{{ evalErr }}</div>

        <div v-if="evalSummary" style="margin-bottom:12px">
          <dl class="kv">
            <template v-for="(v, k) in evalSummary" :key="k">
              <template v-if="typeof v !== 'object'">
                <dt>{{ k }}</dt>
                <dd>{{ v }}</dd>
              </template>
            </template>
          </dl>
        </div>

        <table v-if="evalCases.length">
          <thead>
            <tr>
              <th>case</th><th>intent (want/got)</th><th>intent</th><th>group</th><th>judge</th><th>judge pass</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="c in evalCases" :key="c.case_id || c.id">
              <td class="mono">{{ c.case_id || c.id }}</td>
              <td class="mono">
                {{ c.expected_intent }} → {{ c.predicted_intent || c.actual_intent }}
              </td>
              <td>
                <span :class="['chip', c.intent_correct ? 'success' : 'danger']">
                  {{ c.intent_correct ? 'ok' : 'miss' }}
                </span>
              </td>
              <td>
                <span :class="['chip', c.group_correct ? 'success' : 'danger']">
                  {{ c.group_correct ? 'ok' : 'miss' }}
                </span>
              </td>
              <td>{{ c.judge_score?.toFixed(2) ?? '-' }}</td>
              <td>
                <span :class="['chip', c.judge_pass ? 'success' : 'danger']">
                  {{ c.judge_pass ? 'pass' : 'fail' }}
                </span>
              </td>
            </tr>
          </tbody>
        </table>

        <details v-if="evalResult" style="margin-top:12px">
          <summary>raw evaluation payload</summary>
          <pre>{{ JSON.stringify(evalResult, null, 2) }}</pre>
        </details>
      </div>
    </section>
  </div>
</template>
