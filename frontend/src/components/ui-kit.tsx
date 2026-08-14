import type { ButtonHTMLAttributes, InputHTMLAttributes, LabelHTMLAttributes, ReactNode } from "react";
import { Slot } from "@radix-ui/react-slot";

type ButtonVariant = "primary" | "secondary" | "ghost";
type ButtonSize = "md" | "sm";

export function Button({
  variant = "primary",
  size = "md",
  className = "",
  asChild = false,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: ButtonVariant;
  size?: ButtonSize;
  asChild?: boolean;
}) {
  const base =
    "inline-flex items-center justify-center gap-2 font-medium rounded-md transition-colors disabled:opacity-50 disabled:pointer-events-none focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background";
  const sizes = {
    md: "h-10 px-4 text-sm",
    sm: "h-8 px-3 text-xs",
  } as const;
  const variants = {
    primary: "bg-foreground text-background hover:bg-foreground/90",
    secondary: "bg-background text-foreground border border-border hover:bg-muted",
    ghost: "bg-transparent text-foreground hover:bg-muted",
  } as const;
  const Comp = asChild ? Slot : "button";
  return <Comp className={`${base} ${sizes[size]} ${variants[variant]} ${className}`} {...props} />;
}

export function Input({ className = "", ...props }: InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      className={
        "flex h-10 w-full rounded-md border border-border bg-background px-3 py-2 text-sm shadow-none " +
        "placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 " +
        "focus-visible:ring-ring focus-visible:ring-offset-1 focus-visible:ring-offset-background " +
        "disabled:opacity-50 " +
        className
      }
      {...props}
    />
  );
}

export function Label({ className = "", ...props }: LabelHTMLAttributes<HTMLLabelElement>) {
  return <label className={"text-xs font-medium text-foreground/80 " + className} {...props} />;
}

export function Field({
  label,
  children,
  hint,
}: {
  label: string;
  children: ReactNode;
  hint?: string;
}) {
  return (
    <div className="space-y-1.5">
      <Label>{label}</Label>
      {children}
      {hint && <div className="text-xs text-muted-foreground">{hint}</div>}
    </div>
  );
}

export function Card({ className = "", children }: { className?: string; children: ReactNode }) {
  return (
    <div className={"rounded-xl border border-border bg-card p-6 " + className}>{children}</div>
  );
}

export function Badge({
  children,
  tone = "neutral",
}: {
  children: ReactNode;
  tone?: "neutral" | "muted" | "outline";
}) {
  const tones = {
    neutral: "bg-foreground text-background",
    muted: "bg-muted text-foreground/80",
    outline: "border border-border text-muted-foreground",
  } as const;
  return (
    <span className={`inline-flex items-center rounded-full px-2 py-0.5 text-[11px] font-medium ${tones[tone]}`}>
      {children}
    </span>
  );
}
