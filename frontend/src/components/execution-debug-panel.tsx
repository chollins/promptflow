import type { ReactNode } from "react";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { Card } from "@/components/ui-kit";

export type DebugSection = {
  id: string;
  title: string;
  content: ReactNode;
  defaultOpen?: boolean;
};

export function ExecutionDebugPanel({
  title = "Developer / Execution Details",
  sections,
}: {
  title?: string;
  sections: DebugSection[];
}) {
  const defaultValue = sections.filter((section) => section.defaultOpen).map((section) => section.id);

  return (
    <Card className="p-5">
      <div className="mb-2 text-sm font-medium">{title}</div>
      <Accordion type="multiple" defaultValue={defaultValue} className="w-full">
        {sections.map((section) => (
          <AccordionItem key={section.id} value={section.id}>
            <AccordionTrigger>{section.title}</AccordionTrigger>
            <AccordionContent>{section.content}</AccordionContent>
          </AccordionItem>
        ))}
      </Accordion>
    </Card>
  );
}
