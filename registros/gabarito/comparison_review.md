# Supplied transcription compared with source images

All 12 JPEGs were visually inspected. The canonical `.txt` files preserve the supplied DOCX, including apparent errors. This report is a list of suggested corrections, not a replacement transcription or a claim of complete validation. No full independent AI transcript had been saved before the DOCX arrived. No Kraken recognition run was used for this comparison.

The section numbers map exactly to images 5 through 16. The DOCX's labels `Página 3.-5` through `Página 14.-16` are bibliographic metadata and are omitted from text references. Source images and the DOCX remain untouched. Paragraph indices in `review.json` are zero-based OOXML paragraph indices, not document page numbers.

## Suggested readings

Quoted alternatives are visual readings. High confidence means the discrepancy is clear at the supplied resolution; it does not validate the rest of that page. Medium confidence readings need closer checking. Historical spelling should follow the image, not contemporary Portuguese.

| Image | Supplied GT | Visual reading / issue | Confidence |
|---|---|---|---|
| 5 | `registros`, `Comarca não numeradas`, `secretario o escrevi` | Appears to read `registo`, `Camara são numeradas`, `secretario o lavrei`. Several words in the opening paragraph need checking. | Medium |
| 5 | `Jm P Gonçalves Braga` | Signature is difficult to resolve; keep the supplied expansion separate from a literal handwriting reference until checked. | Low |
| 6 | `recesso eleitoral`, `havido recesso` | `recurso eleitoral`, `havido recurso` | High |
| 6 | `do membro aos membros` | `do membro ou membros` | High |
| 6 | `maneira de votos presentes` | `maioria de votos presentes` | Medium |
| 6 | `defendendo` | Appears to read `observando` before `a Constituição Federal`. | Medium |
| 7 | `sessões de reunião os Vereadores` | `sessões se reunirão os Vereadores` | High |
| 7 | `havendo meio legal` | `havendo numero legal` | High |
| 7 | `As sessões deixarão` | `As sessões durarão` | High |
| 7 | `approvação em desacordos`, `serem observados` | `approvação ou desagrado`, `serem chamados` | Medium |
| 7 | `incube`, `até ter voses` | `incumbe`, `até tres vezes` | High |
| 8 | `que não se prostrar`, `no corrente` | `que não se portar`, `inconveniente` | High |
| 8 | `Requisitar praça publica` | `Requisitar força publica` | High |
| 8 | `marcar`, `marcações`, `marcação` | These terms appear to be `indicações` / `indicação`; inspect the exact singular/plural wording in Artigo 7 and its paragraphs. | Medium |
| 8 | `nos trabalhos municipais` | `nos trabalhos municipaes` | High |
| 8 | `ssesões seguintes` | `sessões seguintes` (the source word continues over the page boundary). | High |
| 9 | `depois de assinadas` | `depois de ouvidas` | High |
| 9 | `provendo ainda crear` | `podendo ainda crear` | High |
| 9 | `Obras Politicas`, `votacção` | `Obras Publicas`, `Instrucção` | High |
| 9 | `será convidado` | `será convocado` | High |
| 10 | `com causa justificada` | `sem causa justificada` - this changes the meaning of Artigo 15. | High |
| 10 | `come-cadas` | `convocadas`, split across handwritten lines. | High |
| 10 | `do Intendente` | Appears to read `do ... Intendente` with an additional word; inspect rather than insert a guess. | Low |
| 11 | `reserva liquida`, `A reserva garantida` | `renda liquida`, `A renda garantida` | High |
| 11 | `registro foi`, `vai confirmar` | `registro fiz`, `vai conforme` | High |
| 11 | `Alas Ferreira`, `Oliveira` | Proper-name readings need review against the handwriting and repeated signatures on later pages; do not silently reconcile them using context. | Medium |
| 12 | `data do anno` | `data do aviso` | High |
| 12 | `força mais`, `firma de direito` | `força maior`, `forma de direito` | High |
| 12 | `prestações normaes` | `prestações mensaes` | High |
| 12 | `data desta leia a seis meses` | `da data desta lei a seis mezes` | High |
| 13 | `reposições em contrário` | `Disposições em contrario` | High |
| 13 | `exgostos` | `exgottos` / `esgottos`; verify the historical spelling. | Medium |
| 13 | `Hygino` and councillor names | Several joined strokes and abbreviated names are ambiguous; the line after `Malfatti` should be reviewed at higher resolution if available. | Low |
| 14 | `perante registro foi` | `presente registro fiz` | High |
| 14 | `publicação desta lei` | The image appears to say `data desta lei` (inspect the clause before changing it). | Medium |
| 14 | `mudança das partes` | `mudança dos postes` | High |
| 14 | `indenisação` | Appears to read `indemnisação`; preserve the image's spelling. | Medium |
| 15 | `Eugenio Augusto Egas` | `Eugenio Andrade Egas` | High |
| 15 | `me representa` | `me reporto` | High |
| 15 | `dispostas taxas` | `disposições` | High |
| 15 | `acesso ou trabalho` | `arrimo ou trabalho` | Medium |
| 15 | Title `Dezembro 5 de 1893` | An overwritten/crossed-out date is visible above the title. Decide consistently whether overwritten text is in scope before scoring. | High |
| 16 | `a lei da obrigação` | `além da obrigação` | High |
| 16 | `reconstruirão` | `reconstruirem` | High |
| 16 | `centimentros` | `centimetros` | High |
| 16 | `servidão en uso publico`, `situacão` in image 15 and several compounds | Distinguish spelling mistakes in the transcription from actual historical forms; whitespace joining/splitting also changes WER. | Medium |

## Scoring consequences

- The supplied text excludes many top-margin rubric signatures and handwritten page numbers. Full-page Kraken may recognize them, producing insertions. The printed watermark is also excluded. Keep the source raw for the baseline; if masking/cropping is later introduced, test it as a distinct operation and document its scope.
- The source sometimes expands abbreviations or joins words differently from the manuscript (`SãoCarlos`, `registros`, names and dates). Strict WER measures agreement with those choices as well as recognition quality.
- Intra-page broken words are retained as supplied, including `come-cadas` and `refe-rem`. A cross-page word such as `ascen-` / `dentes` stays on its original page. Do not move text between references to improve a score.
- Resolve corrections once for the reference set, then evaluate every variant against the same version. The pipeline records reference hashes, source-document hash and scoring configuration. Reviewing a few discrepancies is insufficient to mark a whole page reviewed.
- The listed differences have not been applied. All 12 references remain `supplied_unverified`; WER is available but labeled provisional.
