import { cn } from "@/lib/utils";
import { ComponentProps, FC } from "react";

const Card: FC<ComponentProps<"div">> = ({ className, children, ...props }) => {
  return (
    <div
      className={cn(
        "h-full rounded-lg bg-secondary/20 p-4 border border-white/5",
        "shadow-[0_4px_12px_rgba(0,0,0,0.15)]",
        className
      )}
      {...props}
    >
      {children}
    </div>
  );
};

export default Card; 