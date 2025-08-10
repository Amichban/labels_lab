#!/usr/bin/env python3
"""
Notification hook for approval requests.
Sends notifications to configured channels when Claude needs approval.
"""

import argparse
import json
import os
import subprocess
from datetime import datetime
from typing import Dict, List


class ApprovalNotifier:
    """Handle approval notifications across multiple channels."""
    
    def __init__(self):
        self.config = self.load_config()
    
    def load_config(self) -> Dict:
        """Load notification configuration."""
        config_path = ".claude/notifications.json"
        if os.path.exists(config_path):
            with open(config_path) as f:
                return json.load(f)
        
        # Default configuration
        return {
            "slack": {
                "enabled": True,
                "webhook_url": os.environ.get("SLACK_WEBHOOK_URL"),
                "channel": "#deployments",
                "mention_users": ["@oncall"]
            },
            "github": {
                "enabled": True,
                "create_issue": True,
                "add_comment": True,
                "assign_to": ["team-lead", "oncall"]
            },
            "email": {
                "enabled": False,
                "smtp_server": os.environ.get("SMTP_SERVER"),
                "recipients": ["team@example.com"]
            },
            "pagerduty": {
                "enabled": False,
                "api_key": os.environ.get("PAGERDUTY_API_KEY"),
                "service_id": os.environ.get("PAGERDUTY_SERVICE_ID")
            }
        }
    
    def notify(self, approval_type: str, details: str, channels: List[str]):
        """Send notifications to specified channels."""
        message = self.format_message(approval_type, details)
        
        for channel in channels:
            if channel in self.config and self.config[channel]["enabled"]:
                method = getattr(self, f"notify_{channel}", None)
                if method:
                    try:
                        method(message)
                        print(f"✅ Notification sent to {channel}")
                    except Exception as e:
                        print(f"❌ Failed to notify {channel}: {e}")
    
    def format_message(self, approval_type: str, details: str) -> Dict:
        """Format approval message."""
        return {
            "title": f"🔔 Approval Required: {approval_type}",
            "timestamp": datetime.now().isoformat(),
            "type": approval_type,
            "details": details,
            "actions": [
                {"label": "Approve", "url": self.get_approval_url(approval_type)},
                {"label": "Reject", "url": self.get_rejection_url(approval_type)},
                {"label": "View Details", "url": self.get_details_url(approval_type)}
            ]
        }
    
    def notify_slack(self, message: Dict):
        """Send Slack notification."""
        webhook_url = self.config["slack"]["webhook_url"]
        if not webhook_url:
            raise ValueError("Slack webhook URL not configured")
        
        slack_message = {
            "text": message["title"],
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": message["title"]
                    }
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": message["details"]
                    }
                },
                {
                    "type": "actions",
                    "elements": [
                        {
                            "type": "button",
                            "text": {"type": "plain_text", "text": action["label"]},
                            "url": action["url"],
                            "style": "primary" if action["label"] == "Approve" else "default"
                        }
                        for action in message["actions"]
                    ]
                }
            ]
        }
        
        # Send to Slack
        subprocess.run([
            "curl", "-X", "POST",
            "-H", "Content-Type: application/json",
            "-d", json.dumps(slack_message),
            webhook_url
        ], check=True)
    
    def notify_github(self, message: Dict):
        """Create GitHub issue or comment."""
        if self.config["github"]["create_issue"]:
            # Create issue using gh CLI
            title = message["title"]
            body = f"""
{message["details"]}

## Actions Required
- [ ] Review the changes
- [ ] Approve or reject the request
- [ ] Monitor after approval

## Links
{chr(10).join(f"- [{action['label']}]({action['url']})" for action in message["actions"])}

---
*Created by Claude Approval Bot at {message["timestamp"]}*
"""
            
            # Create issue
            result = subprocess.run([
                "gh", "issue", "create",
                "--title", title,
                "--body", body,
                "--label", "approval-required",
                "--assignee", ",".join(self.config["github"]["assign_to"])
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                issue_url = result.stdout.strip()
                print(f"GitHub issue created: {issue_url}")
    
    def notify_email(self, message: Dict):
        """Send email notification."""
        # Implementation depends on SMTP configuration
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart
        
        smtp_server = self.config["email"]["smtp_server"]
        recipients = self.config["email"]["recipients"]
        
        if not smtp_server:
            raise ValueError("SMTP server not configured")
        
        # Create email
        msg = MIMEMultipart()
        msg["Subject"] = message["title"]
        msg["From"] = "claude@example.com"
        msg["To"] = ", ".join(recipients)
        
        # Email body
        body = f"""
{message['details']}

Actions:
{chr(10).join(f"- {action['label']}: {action['url']}" for action in message['actions'])}

---
Sent at {message['timestamp']}
"""
        msg.attach(MIMEText(body, "plain"))
        
        # Send email
        with smtplib.SMTP(smtp_server) as server:
            server.send_message(msg)
    
    def notify_pagerduty(self, message: Dict):
        """Trigger PagerDuty alert."""
        api_key = self.config["pagerduty"]["api_key"]
        service_id = self.config["pagerduty"]["service_id"]
        
        if not api_key or not service_id:
            raise ValueError("PagerDuty not configured")
        
        # Create PagerDuty incident
        incident = {
            "incident": {
                "type": "incident",
                "title": message["title"],
                "service": {"id": service_id, "type": "service_reference"},
                "body": {"type": "incident_body", "details": message["details"]},
                "urgency": "high" if "production" in message["type"].lower() else "low"
            }
        }
        
        subprocess.run([
            "curl", "-X", "POST",
            "https://api.pagerduty.com/incidents",
            "-H", f"Authorization: Token token={api_key}",
            "-H", "Content-Type: application/json",
            "-d", json.dumps(incident)
        ], check=True)
    
    def get_approval_url(self, approval_type: str) -> str:
        """Get URL for approval action."""
        base_url = os.environ.get("APPROVAL_BASE_URL", "https://approvals.example.com")
        return f"{base_url}/approve/{approval_type}"
    
    def get_rejection_url(self, approval_type: str) -> str:
        """Get URL for rejection action."""
        base_url = os.environ.get("APPROVAL_BASE_URL", "https://approvals.example.com")
        return f"{base_url}/reject/{approval_type}"
    
    def get_details_url(self, approval_type: str) -> str:
        """Get URL for viewing details."""
        if "flag" in approval_type:
            return f"https://github.com/{os.environ.get('GITHUB_REPOSITORY', 'org/repo')}/blob/main/config/flags.yaml"
        return f"https://github.com/{os.environ.get('GITHUB_REPOSITORY', 'org/repo')}"


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Send approval notifications")
    parser.add_argument("--type", required=True, help="Type of approval needed")
    parser.add_argument("--details", required=True, help="Details of the approval request")
    parser.add_argument("--channels", required=True, help="Comma-separated list of channels")
    parser.add_argument("--flag", help="Feature flag name (if applicable)")
    parser.add_argument("--pr", help="PR URL (if applicable)")
    parser.add_argument("--environment", help="Deployment environment (if applicable)")
    
    args = parser.parse_args()
    
    # Parse channels
    channels = args.channels.split(",")
    
    # Build details
    details = args.details
    if args.flag:
        details += f"\n\n**Feature Flag**: `{args.flag}`"
    if args.pr:
        details += f"\n\n**Pull Request**: {args.pr}"
    if args.environment:
        details += f"\n\n**Environment**: {args.environment}"
    
    # Send notifications
    notifier = ApprovalNotifier()
    notifier.notify(args.type, details, channels)


if __name__ == "__main__":
    main()