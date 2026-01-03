"use client"

import { useState, useEffect } from "react"
import Link from "next/link"
import { usePathname, useRouter } from "next/navigation"
import { cn } from "@/lib/utils"
import { useAuth } from "@/lib/auth"
import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import { ScrollArea } from "@/components/ui/scroll-area"
import {
  LayoutDashboard,
  Database,
  Plus,
  Key,
  Globe,
  Settings,
  LogOut,
  ChevronDown,
  ChevronRight,
  Folder,
  Shield,
  Building2,
  CreditCard,
  BarChart3
} from "lucide-react"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"

interface SidebarProps {
  children: React.ReactNode
}

interface Playground {
  id: string
  name: string
}

export function Sidebar({ children }: SidebarProps) {
  const pathname = usePathname()
  const router = useRouter()
  const { user, logout } = useAuth()
  const [playgrounds, setPlaygrounds] = useState<Playground[]>([])
  const [playgroundsOpen, setPlaygroundsOpen] = useState(true)

  useEffect(() => {
    fetchPlaygrounds()
  }, [])

  const fetchPlaygrounds = async () => {
    try {
      const res = await fetch("/api/playgrounds", { credentials: "include" })
      if (!res.ok) return
      const data = await res.json()
      setPlaygrounds(Array.isArray(data) ? data : [])
    } catch (e) { console.error(e) }
  }

  const createPlayground = async () => {
    try {
      const res = await fetch("/api/playgrounds", {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: "New Playground" })
      })
      const data = await res.json()
      if (data.id) {
        fetchPlaygrounds()
        router.push(`/playground/${data.id}`)
      }
    } catch (e) { console.error(e) }
  }

  const navItems = [
    { icon: LayoutDashboard, label: "Overview", href: "/" },
    { icon: Database, label: "Data Sources", href: "/data-sources" },
  ]

  const configItems = [
    { icon: Key, label: "API Keys", href: "/api-keys" },
    { icon: Globe, label: "Endpoints", href: "/endpoints" },
    { icon: Settings, label: "Settings", href: "/settings" },
    { icon: Shield, label: "Admin", href: "/admin", adminOnly: true },
  ]

  const getInitials = (email: string) => email?.slice(0, 2).toUpperCase() || "U"

  return (
    <div className="flex h-screen bg-background">
      <div className="w-56 border-r flex flex-col bg-muted/30">
        {/* Logo */}
        <div className="p-3 border-b">
          <Link href="/" className="flex items-center gap-2 p-2">
            <div className="h-8 w-8 rounded-lg bg-primary text-primary-foreground flex items-center justify-center font-bold text-sm">
              SL
            </div>
            <div>
              <p className="text-sm font-semibold">Schema Labs</p>
              <p className="text-xs text-muted-foreground">Pro</p>
            </div>
          </Link>
        </div>

        <ScrollArea className="flex-1">
          <div className="p-3 space-y-6">
            {/* Navigation */}
            <div className="space-y-1">
              {navItems.map((item) => (
                <Link
                  key={item.href}
                  href={item.href}
                  className={cn(
                    "flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-colors",
                    pathname === item.href
                      ? "bg-primary text-primary-foreground"
                      : "text-muted-foreground hover:text-foreground hover:bg-muted"
                  )}
                >
                  <item.icon className="h-4 w-4" />
                  {item.label}
                </Link>
              ))}
            </div>

            {/* Playgrounds */}
            <div>
              <button
                onClick={() => setPlaygroundsOpen(!playgroundsOpen)}
                className="flex items-center justify-between w-full px-3 py-2 text-xs font-medium text-muted-foreground hover:text-foreground"
              >
                <span>Playground</span>
                {playgroundsOpen ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
              </button>
              {playgroundsOpen && (
                <div className="mt-1 space-y-1">
                  <button
                    onClick={createPlayground}
                    className="flex items-center gap-3 px-3 py-2 w-full rounded-lg text-sm text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
                  >
                    <Plus className="h-4 w-4" />
                    New
                  </button>
                  {playgrounds.map((pg) => (
                    <Link
                      key={pg.id}
                      href={`/playground/${pg.id}`}
                      className={cn(
                        "flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-colors",
                        pathname === `/playground/${pg.id}`
                          ? "bg-primary text-primary-foreground"
                          : "text-muted-foreground hover:text-foreground hover:bg-muted"
                      )}
                    >
                      <Folder className="h-4 w-4" />
                      <span className="truncate">{pg.name}</span>
                    </Link>
                  ))}
                </div>
              )}
            </div>

            {/* Configuration */}
            <div>
              <p className="px-3 py-2 text-xs font-medium text-muted-foreground">Configuration</p>
              <div className="space-y-1">
                {configItems
                  .filter(item => !item.adminOnly || user?.role === "admin")
                  .map((item) => (
                    <Link
                      key={item.href}
                      href={item.href}
                      className={cn(
                        "flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-colors",
                        pathname === item.href
                          ? "bg-primary text-primary-foreground"
                          : "text-muted-foreground hover:text-foreground hover:bg-muted"
                      )}
                    >
                      <item.icon className="h-4 w-4" />
                      {item.label}
                    </Link>
                  ))}
              </div>
            </div>
          </div>
        </ScrollArea>

        {/* User Dropdown */}
        <div className="p-3 border-t">
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <button className="w-full flex items-center gap-3 p-2 rounded-lg hover:bg-muted transition-colors">
                <Avatar className="h-8 w-8">
                  <AvatarFallback className="text-xs">{getInitials(user?.email || "")}</AvatarFallback>
                </Avatar>
                <div className="flex-1 text-left min-w-0">
                  <p className="text-sm font-medium truncate">{user?.name || user?.email?.split("@")[0]}</p>
                  <p className="text-xs text-muted-foreground truncate">{user?.email}</p>
                </div>
                <ChevronDown className="h-4 w-4 text-muted-foreground shrink-0" />
              </button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="start" side="top" className="w-52">
              <DropdownMenuItem asChild>
                <Link href="/account" className="cursor-pointer">
                  <Building2 className="h-4 w-4 mr-2" />
                  Account
                </Link>
              </DropdownMenuItem>
              <DropdownMenuItem asChild>
                <Link href="/organizations" className="cursor-pointer">
                  <Building2 className="h-4 w-4 mr-2" />
                  Organizations
                </Link>
              </DropdownMenuItem>
              <DropdownMenuItem asChild>
                <Link href="/billing" className="cursor-pointer">
                  <CreditCard className="h-4 w-4 mr-2" />
                  Billing
                </Link>
              </DropdownMenuItem>
              <DropdownMenuItem asChild>
                <Link href="/usage" className="cursor-pointer">
                  <BarChart3 className="h-4 w-4 mr-2" />
                  Usage
                </Link>
              </DropdownMenuItem>
              <DropdownMenuItem onClick={logout} className="text-destructive cursor-pointer">
                <LogOut className="h-4 w-4 mr-2" />
                Log out
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {children}
      </div>
    </div>
  )
}
