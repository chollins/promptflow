import { createFileRoute, Link } from "@tanstack/react-router";
import { Sparkles, ArrowRight } from "lucide-react";
import { Button } from "@/components/ui-kit";

export const Route = createFileRoute("/")({
  component: Landing,
});

function Landing() {
  return (
    <div className="min-h-screen flex flex-col bg-background text-foreground">
      <header className="h-16 border-b border-border">
        <div className="max-w-6xl mx-auto h-full px-6 flex items-center justify-between">
          <Link to="/" className="flex items-center gap-2 font-semibold tracking-tight">
            <Sparkles className="h-4 w-4" strokeWidth={2.25} />
            PromptFlow
          </Link>
          <div className="flex items-center gap-2">
            <Link to="/login">
              <Button variant="ghost" size="sm">Sign in</Button>
            </Link>
            <Link to="/create-organization">
              <Button size="sm">Get started</Button>
            </Link>
          </div>
        </div>
      </header>

      <main className="flex-1 flex items-center">
        <div className="max-w-3xl mx-auto px-6 py-24 md:py-32 text-center">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-border text-xs text-muted-foreground mb-8">
            <span className="h-1.5 w-1.5 rounded-full bg-foreground" />
            Multi-tenant AI platform
          </div>
          <h1 className="text-5xl md:text-7xl font-semibold tracking-tight leading-[1.05]">
            PromptFlow
          </h1>
          <p className="mt-6 text-lg md:text-xl text-muted-foreground max-w-xl mx-auto leading-relaxed">
            Build AI workflows for your organization. Invite your team, share prompts,
            and ship faster with structured flows.
          </p>
          <div className="mt-10 flex justify-center">
            <Link to="/login">
              <Button>
                Sign In
                <ArrowRight className="h-4 w-4" />
              </Button>
            </Link>
          </div>
        </div>
      </main>

      <footer className="border-t border-border">
        <div className="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between text-xs text-muted-foreground">
          <div>© {new Date().getFullYear()} PromptFlow</div>
          <div className="flex gap-6">
            <a href="#" className="hover:text-foreground">Privacy</a>
            <a href="#" className="hover:text-foreground">Terms</a>
          </div>
        </div>
      </footer>
    </div>
  );
}
