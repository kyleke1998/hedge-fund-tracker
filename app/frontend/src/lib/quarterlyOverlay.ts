/**
 * Merge a quarterly series onto a price series as a step function.
 *
 * Quarterly facts (13F holdings, the cost-basis band) are known only as of a
 * quarter end, while the price series is daily, weekly or monthly depending on
 * the range on screen. Each price point therefore carries the most recent
 * quarter whose as-of date it has passed — the fact stood, as far as the
 * filings say, until the next quarter restated it. Points earlier than the
 * first quarter on record get nothing, so an overlay starts where its data
 * does instead of being anchored to a fabricated zero.
 */
export function mergeQuarterly<T extends { date: string }, P extends { asOf: string }, F>(
  series: readonly T[],
  quarterly: readonly P[],
  project: (point: P) => F,
): (T & Partial<F>)[] {
  // The spread alone is a `T`; the merged type only widens it with optional fields.
  const untouched = (point: T) => ({ ...point }) as T & Partial<F>;
  if (quarterly.length === 0) return series.map(untouched);

  const ordered = [...quarterly].sort((a, b) => a.asOf.localeCompare(b.asOf));
  let cursor = -1;

  return series.map((point) => {
    while (cursor + 1 < ordered.length && ordered[cursor + 1].asOf <= point.date) cursor += 1;
    if (cursor < 0) return untouched(point);
    return { ...point, ...project(ordered[cursor]) };
  });
}
