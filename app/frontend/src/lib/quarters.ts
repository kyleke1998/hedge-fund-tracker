/**
 * Branded type representing a valid quarter identifier in the form "YYYYQ[1-4]".
 * Use isQuarter() or assertQuarter() to narrow strings from untyped sources (API, CSV).
 */
export type Quarter = `${number}Q${1 | 2 | 3 | 4}`;

const QUARTER_RE = /^\d{4}Q[1-4]$/;

export function isQuarter(value: string): value is Quarter {
  return QUARTER_RE.test(value);
}

export function assertQuarter(value: string): Quarter {
  if (!isQuarter(value)) {
    throw new Error(`Invalid quarter: ${value}`);
  }
  return value;
}

/**
 * Filter and sort a list of untyped strings into valid Quarters.
 */
export function parseQuarters(values: readonly string[]): readonly Quarter[] {
  return values.filter(isQuarter).sort() as readonly Quarter[];
}

/**
 * The calendar end date of a quarter, as an ISO `YYYY-MM-DD` string.
 *
 * This is the "as of" date a 13F position describes (the filing itself lands up
 * to 45 days later), so it is the date holdings are anchored to on a price
 * timeline. Built from string parts rather than `Date` so the result never
 * shifts with the viewer's timezone.
 */
export function quarterEndDate(quarter: Quarter): string {
  const ENDS = { "1": "03-31", "2": "06-30", "3": "09-30", "4": "12-31" } as const;
  const year = quarter.slice(0, 4);
  const q = quarter.slice(5) as keyof typeof ENDS;
  return `${year}-${ENDS[q]}`;
}
