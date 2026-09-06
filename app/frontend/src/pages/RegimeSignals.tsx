import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip as RechartsTooltip,
  XAxis,
  YAxis,
  type TooltipContentProps,
} from "recharts";
import { Activity } from "lucide-react";

import { EmptyState } from "@/components/ui/EmptyState";
import { LoadingState } from "@/components/ui/LoadingState";
import { InfoTooltip } from "@/components/ui/InfoTooltip";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { getRegime, reindexShareIndex, sectorWindowChange } from "@/lib/data/regime";

const RISK_OFF = "hsl(14, 62%, 43%)";
const RISK_ON = "hsl(168, 100%, 25%)";
const NEUTRAL = "hsl(230, 46%, 45%)";

const quarterLabel = (quarter: string) => quarter.replace("Q", " Q");

/** Chart series where a value is missing for a quarter must break, not read as 0. */
type Point = Record<string, string | number | null>;

function ChartTooltip({
  active,
  label,
  payload,
  unit = "",
}: TooltipContentProps & { unit?: string }) {
  if (!active || !payload?.length) return null;
  const rows = payload.flatMap((entry) =>
    typeof entry.name === "string"
      ? [{ name: entry.name, value: entry.value, color: entry.color }]
      : [],
  );
  if (!rows.length) return null;
  return (
    <div className="rounded-lg border border-border bg-popover px-3 py-2 text-xs shadow-lg">
      <div className="mb-1 text-muted-foreground">{label}</div>
      {rows.map((row) => (
        <div key={row.name} className="flex justify-between gap-4 font-mono font-semibold">
          <span style={{ color: row.color }}>{row.name}</span>
          <span>{typeof row.value === "number" ? `${row.value.toFixed(2)}${unit}` : "—"}</span>
        </div>
      ))}
    </div>
  );
}

/**
 * Recharts `content` renderer for a unit suffix. A typed factory rather than an
 * inline arrow, so the props parameter is never implicitly `any` and each chart
 * keeps a stable callback identity across renders.
 */
const tooltipFor =
  (unit = "") =>
  (props: TooltipContentProps) => <ChartTooltip {...props} unit={unit} />;

const PLAIN_TOOLTIP = tooltipFor();
const PERCENT_TOOLTIP = tooltipFor("%");
const POINTS_TOOLTIP = tooltipFor(" pts");

function Panel({
  title,
  caption,
  info,
  children,
  action,
}: {
  title: string;
  caption: string;
  info?: string;
  children: React.ReactNode;
  action?: React.ReactNode;
}) {
  return (
    <section className="surface p-4 sm:p-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="section-title flex items-center gap-1.5">
            {title}
            {info && <InfoTooltip text={info} />}
          </h2>
          <p className="mt-1 text-sm text-muted-foreground">{caption}</p>
        </div>
        {action}
      </div>
      <div className="mt-4">{children}</div>
    </section>
  );
}

/** Quarter dropdown. Declared at module scope so the Select is never remounted. */
function QuarterPicker({
  value,
  onChange,
  label,
  quarters,
}: {
  value: string;
  onChange: (value: string) => void;
  label: string;
  quarters: string[];
}) {
  return (
    <Select value={value} onValueChange={onChange}>
      <SelectTrigger className="h-8 w-[124px] text-xs" aria-label={label}>
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        {quarters.map((quarter) => (
          <SelectItem key={quarter} value={quarter} className="text-xs">
            {quarterLabel(quarter)}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}

/**
 * A single regime dial over time, with a dashed reference at the level that
 * separates expansion from contraction.
 */
function DialChart({
  data,
  label,
  reference,
  unit,
}: {
  data: Point[];
  label: string;
  reference?: number;
  unit?: string;
}) {
  return (
    <ResponsiveContainer width="100%" height={200}>
      <LineChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -18 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" vertical={false} />
        <XAxis
          dataKey="quarter"
          tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
          tickLine={false}
          axisLine={false}
          interval="preserveStartEnd"
          minTickGap={16}
        />
        <YAxis
          tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
          tickLine={false}
          axisLine={false}
          width={44}
        />
        {reference !== undefined && (
          <ReferenceLine
            y={reference}
            stroke="hsl(var(--muted-foreground))"
            strokeDasharray="4 4"
          />
        )}
        <RechartsTooltip content={unit === "%" ? PERCENT_TOOLTIP : PLAIN_TOOLTIP} />
        <Line
          type="monotone"
          dataKey="value"
          name={label}
          stroke={NEUTRAL}
          strokeWidth={2}
          dot={{ r: 2.5 }}
          activeDot={{ r: 5 }}
          connectNulls={false}
          isAnimationActive={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}

export default function RegimeSignals() {
  const { data, isLoading, isError } = useQuery({
    queryKey: ["regime"],
    queryFn: getRegime,
  });

  const [base, setBase] = useState<string | null>(null);
  const [from, setFrom] = useState<string | null>(null);
  const [to, setTo] = useState<string | null>(null);

  const quarters = data?.quarters ?? [];
  // Quarters run oldest-first, so the mega-cap index defaults to the start of the
  // record and the sector window to the most recent quarter-over-quarter move.
  const baseQuarter = base ?? quarters[0] ?? "";
  const fromQuarter = from ?? quarters[quarters.length - 2] ?? "";
  const toQuarter = to ?? quarters[quarters.length - 1] ?? "";

  const dials = useMemo(() => {
    if (!data) return null;
    const series = (key: string): Point[] =>
      data.quarters.map((q, i) => ({
        quarter: quarterLabel(q),
        value: data.metrics[key]?.[i] ?? null,
      }));
    return {
      newClose: series("new_close"),
      recycle: series("recycle"),
      untouched: series("untouched_pct"),
    };
  }, [data]);

  const megaSeries = useMemo(() => {
    if (!data) return [];
    const shares = reindexShareIndex(data, baseQuarter);
    const baseIndex = data.quarters.indexOf(baseQuarter);
    const weightBase = baseIndex >= 0 ? data.mega.weightPct[baseIndex] : null;
    return data.quarters.map((q, i) => {
      const weight = data.mega.weightPct[i];
      return {
        quarter: quarterLabel(q),
        Weight: weight !== null && weightBase ? (weight / weightBase) * 100 : null,
        Shares: shares[i],
      };
    });
  }, [data, baseQuarter]);

  const sectorMoves = useMemo(
    () => (data ? sectorWindowChange(data, fromQuarter, toQuarter) : []),
    [data, fromQuarter, toQuarter],
  );

  if (isLoading) return <LoadingState message="Loading regime signals…" />;
  if (isError || !data || !dials || quarters.length === 0) {
    return (
      <EmptyState
        icon={Activity}
        title="Regime signals unavailable"
        description="Run `pipenv run gen-regime` to build database/regime.csv from the quarterly filings."
      />
    );
  }

  return (
    <div className="space-y-6 max-w-screen-2xl">
      <div>
        <span className="eyebrow">Positioning over time</span>
        <h1 className="page-title mt-1.5">
          <Activity className="page-title-icon" /> Regime Signals
        </h1>
        <p className="mt-1.5 text-sm text-muted-foreground">
          How the tracked funds shift posture across market cycles — {quarters.length} quarters,
          measured in position and share counts rather than dollars, so a market move is never
          mistaken for a decision.
        </p>
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        <Panel
          title="New : Closed positions"
          caption="Appetite for fresh ideas. Above 1.0 the book is widening."
          info="Positions opened divided by positions closed, counted on the funds present in both quarters so a growing tracked universe never reads as buying."
        >
          <DialChart data={dials.newClose} label="New : Closed" reference={1} />
        </Panel>
        <Panel
          title="Capital recycling"
          caption="Dollars into new names ÷ dollars released by exits."
          info="Below 1.0 the book is shrinking: exits release more capital than new positions absorb. In the 2022 drawdown it bottomed at 0.58 in Q4, the quarter containing the price low."
        >
          <DialChart data={dials.recycle} label="Recycling" reference={1} />
        </Panel>
        <Panel
          title="Untouched positions"
          caption="Share of positions left completely unchanged."
          info="A paralysis gauge. In the 2022 drawdown it peaked near 23% in Q3, the least active quarter, then fell as funds repositioned through the 2023 base."
        >
          <DialChart data={dials.untouched} label="Untouched" unit="%" />
        </Panel>
      </div>

      <Panel
        title="Mega-cap exposure: weight versus shares held"
        caption={`MSFT, AMZN, NVDA, META, GOOGL, AAPL, TSLA, AMD — both indexed to ${quarterLabel(baseQuarter)} = 100.`}
        info="Shares are split-adjusted and chain-linked on funds common to consecutive quarters, so neither a stock split nor a fund joining the roster can move the line. When weight rises while shares fall, the exposure came from price, not from buying."
        action={
          <div className="flex items-center gap-2">
            <span className="text-xs text-muted-foreground">Base</span>
            <QuarterPicker
              value={baseQuarter}
              onChange={setBase}
              label="Index base quarter"
              quarters={quarters}
            />
          </div>
        }
      >
        <div className="mb-3 flex flex-wrap gap-4 text-xs text-muted-foreground">
          <span className="inline-flex items-center gap-1.5">
            <i className="h-2.5 w-2.5 rounded-sm" style={{ background: RISK_OFF }} /> Portfolio
            weight
          </span>
          <span className="inline-flex items-center gap-1.5">
            <i className="h-2.5 w-2.5 rounded-sm" style={{ background: RISK_ON }} /> Shares held
            (split-adjusted)
          </span>
        </div>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={megaSeries} margin={{ top: 8, right: 12, bottom: 0, left: -14 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" vertical={false} />
            <XAxis
              dataKey="quarter"
              tick={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }}
              tickLine={false}
              axisLine={false}
              interval="preserveStartEnd"
              minTickGap={16}
            />
            <YAxis
              tick={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }}
              tickLine={false}
              axisLine={false}
              width={46}
            />
            <ReferenceLine y={100} stroke="hsl(var(--muted-foreground))" strokeDasharray="4 4" />
            <RechartsTooltip content={PLAIN_TOOLTIP} />
            <Line
              type="monotone"
              dataKey="Weight"
              stroke={RISK_OFF}
              strokeWidth={2.5}
              dot={{ r: 2.5 }}
              activeDot={{ r: 5 }}
              connectNulls={false}
              isAnimationActive={false}
            />
            <Line
              type="monotone"
              dataKey="Shares"
              stroke={RISK_ON}
              strokeWidth={2.5}
              dot={{ r: 2.5 }}
              activeDot={{ r: 5 }}
              connectNulls={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </Panel>

      <Panel
        title="Sector weight change"
        caption={`Percentage points of classified portfolio value, ${quarterLabel(fromQuarter)} → ${quarterLabel(toQuarter)}.`}
        info="Levels are shares of each quarter's own classified value, so a change over a long window partly reflects a growing tracked universe as well as rotation. Unclassified value is excluded, never bucketed."
        action={
          <div className="flex items-center gap-2">
            <QuarterPicker
              value={fromQuarter}
              onChange={setFrom}
              label="Window start quarter"
              quarters={quarters}
            />
            <span className="text-xs text-muted-foreground">→</span>
            <QuarterPicker
              value={toQuarter}
              onChange={setTo}
              label="Window end quarter"
              quarters={quarters}
            />
          </div>
        }
      >
        {sectorMoves.length === 0 ? (
          <EmptyState title="No overlapping sector data for this window" padding="sm" />
        ) : (
          <ResponsiveContainer width="100%" height={Math.max(260, sectorMoves.length * 30)}>
            <BarChart
              data={sectorMoves}
              layout="vertical"
              margin={{ top: 4, right: 28, bottom: 4, left: 8 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" horizontal={false} />
              <XAxis
                type="number"
                tick={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }}
                tickLine={false}
                axisLine={false}
              />
              <YAxis
                type="category"
                dataKey="sector"
                width={140}
                tick={{ fontSize: 11, fill: "hsl(var(--foreground))" }}
                tickLine={false}
                axisLine={false}
              />
              <ReferenceLine x={0} stroke="hsl(var(--border))" />
              <RechartsTooltip
                cursor={{ fill: "hsl(var(--muted) / 0.35)" }}
                content={POINTS_TOOLTIP}
              />
              <Bar dataKey="delta" name="Change" radius={[2, 2, 2, 2]} isAnimationActive={false}>
                {sectorMoves.map((move) => (
                  <Cell key={move.sector} fill={move.delta >= 0 ? RISK_ON : RISK_OFF} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        )}
      </Panel>
    </div>
  );
}
