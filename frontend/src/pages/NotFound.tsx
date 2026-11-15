import { Link } from "react-router-dom";
import { Empty } from "../components/ui";

export default function NotFound() {
  return (
    <main className="page">
      <div className="panel">
        <Empty title="This page doesn't exist" action={<Link className="btn btn-primary" to="/">Go to the overview</Link>}>
          The link may be old, or the form was deleted from the history.
        </Empty>
      </div>
    </main>
  );
}
