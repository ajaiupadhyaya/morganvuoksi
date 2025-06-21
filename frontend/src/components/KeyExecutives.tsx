import { KeyExecutive } from "@/types";
import { FC } from "react";
import Card from "./ui/Card";
import { Briefcase } from "lucide-react";

interface KeyExecutivesProps {
  executives: KeyExecutive[];
}

const KeyExecutives: FC<KeyExecutivesProps> = ({ executives }) => {
  return (
    <Card>
      <h2 className="mb-4 flex items-center gap-2 font-semibold text-secondary-foreground">
        <Briefcase size={18} />
        Key Executives
      </h2>
      <ul className="space-y-3 text-sm">
        {executives.map((exec) => (
          <li key={exec.name} className="flex items-baseline justify-between">
            <span>{exec.name}</span>
            <span className="text-right text-muted-foreground">
              {exec.title}
            </span>
          </li>
        ))}
      </ul>
    </Card>
  );
};

export default KeyExecutives; 