import { Headline } from "@/types";
import { FC } from "react";
import Card from "./ui/Card";
import { Newspaper } from "lucide-react";

interface HeadlinesProps {
  headlines: Headline[];
}

const Headlines: FC<HeadlinesProps> = ({ headlines }) => {
  return (
    <Card>
      <h2 className="mb-4 flex items-center gap-2 font-semibold text-secondary-foreground">
        <Newspaper size={18} />
        Headlines
      </h2>
      <ul className="space-y-4 text-sm">
        {headlines.map((item) => (
          <li key={item.title}>
            <p className="font-medium hover:text-accent-blue cursor-pointer">
              {item.title}
            </p>
            <p className="text-xs text-muted-foreground">{item.source}</p>
          </li>
        ))}
      </ul>
    </Card>
  );
};

export default Headlines; 