"use client"

import type React from "react"
import { useState, createContext, useContext, useEffect, useRef } from "react"
import Link from "next/link"
import Image from "next/image"
import { usePathname, useRouter } from "next/navigation"
import { cn } from "@/lib/utils"
import { useAuth } from "@/lib/auth"
import {
  LayoutDashboard,
  Database,
  Cpu,
  MessageSquare,
  Settings,
  HelpCircle,
  Layers,
  PanelLeftClose,
  PanelLeft,
  Sun,
  Moon,
  ChevronDown,
  ChevronRight,
  Plus,
  MoreHorizontal,
  Pencil,
  Trash2,
  Check,
  X,
  User,
  CreditCard,
  BarChart3,
  LogOut,
  Mail,
  Calendar,
  Loader2,
} from "lucide-react"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import type { PlaygroundSession } from "@/lib/types"

const navigation = [
  { name: "Dashboard", href: "/", icon: LayoutDashboard },
  { name: "Database", href: "/datasets", icon: Database },
  { name: "Model Builder", href: "/build", icon: Cpu },
  { name: "Models", href: "/models", icon: Layers },
]

interface BackendQuery {
  id: string
  name: string
  createdAt: string
  model?: string
}

export const SidebarContext = createContext<{
  collapsed: boolean
  setCollapsed: (value: boolean) => void
  theme: "dark" | "light"
  setTheme: (value: "dark" | "light") => void
  chatSessions: PlaygroundSession[]
  setChatSessions: React.Dispatch<React.SetStateAction<PlaygroundSession[]>>
  addChatSession: (session: PlaygroundSession) => void
  resetPlayground: () => void
}>({
  collapsed: false,
  setCollapsed: () => {},
  theme: "dark",
  setTheme: () => {},
  chatSessions: [],
  setChatSessions: () => {},
  addChatSession: () => {},
  resetPlayground: () => {},
})

export function useSidebar() {
  return useContext(SidebarContext)
}

export function SidebarProvider({ children }: { children: React.ReactNode }) {
  const [collapsed, setCollapsed] = useState(false)
  const [theme, setTheme] = useState<"dark" | "light">("dark")
  const [chatSessions, setChatSessions] = useState<PlaygroundSession[]>([])
  const [mounted, setMounted] = useState(false)
  const [queriesLoaded, setQueriesLoaded] = useState(false)

  useEffect(() => {
    const savedTheme = localStorage.getItem("schemalabs-theme") as "dark" | "light" | null
    if (savedTheme) {
      setTheme(savedTheme)
    }
    setMounted(true)
  }, [])

  // Load queries from backend
  useEffect(() => {
    const loadQueries = async () => {
      try {
        const res = await fetch("/api/queries", { credentials: "include" })
        if (res.ok) {
          const data = await res.json()
          if (data.queries && Array.isArray(data.queries)) {
            const sessions: PlaygroundSession[] = data.queries
              .filter((q: BackendQuery) => {
                // Filter out auto-created/hidden queries
                const name = q.name || ""
                // Hidden queries (model-based chats)
                if (name.startsWith('__hidden__')) return false
                // Model-named queries
                if (name.startsWith('model_') || name.startsWith('Model_') || name.includes('_merged_')) return false
                // Prompt-text queries (lowercase start with spaces)
                if (/^[a-z]/.test(name) && name.includes(' ')) return false
                return true
              })
              .map((q: BackendQuery) => ({
                id: q.id,
                name: q.name || "Untitled Chat",
                modelIds: [],
                llmIds: ["claude"],
                messages: [],
                createdAt: new Date(q.createdAt),
                updatedAt: new Date(q.createdAt),
              }))
            setChatSessions(sessions)
          }
        }
      } catch (e) {
        console.error("Failed to load queries:", e)
      }
      setQueriesLoaded(true)
    }
    loadQueries()
  }, [])

  useEffect(() => {
    if (!mounted) return
    
    localStorage.setItem("schemalabs-theme", theme)
    
    if (theme === "dark") {
      document.documentElement.classList.add("dark")
      document.body.style.backgroundColor = "#0A0A0B"
      document.body.style.color = "#ffffff"
    } else {
      document.documentElement.classList.remove("dark")
      document.body.style.backgroundColor = "#f8f9fa"
      document.body.style.color = "#1a1a1a"
    }
  }, [theme, mounted])

  const addChatSession = (session: PlaygroundSession) => {
    setChatSessions((prev) => [session, ...prev])
    
    // Save to backend
    fetch("/api/queries/create", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "include",
      body: JSON.stringify({
        name: session.name,
        model: "gpt-4o",
      }),
    }).catch(e => console.error("Failed to save query:", e))
  }

  const resetPlayground = () => {}

  return (
    <SidebarContext.Provider
      value={{
        collapsed,
        setCollapsed,
        theme,
        setTheme,
        chatSessions,
        setChatSessions,
        addChatSession,
        resetPlayground,
      }}
    >
      {children}
    </SidebarContext.Provider>
  )
}

function ChatSessionItem({ session, theme }: { session: PlaygroundSession; theme: "dark" | "light" }) {
  const { chatSessions, setChatSessions } = useSidebar()
  const [isEditing, setIsEditing] = useState(false)
  const [editName, setEditName] = useState(session.name)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (isEditing && inputRef.current) {
      inputRef.current.focus()
      inputRef.current.select()
    }
  }, [isEditing])

  const handleRename = () => {
    if (editName.trim()) {
      setChatSessions(chatSessions.map((s) => (s.id === session.id ? { ...s, name: editName.trim() } : s)))
      
      // Update on backend
      fetch("/api/queries/update", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ id: session.id, name: editName.trim() }),
      }).catch(e => console.error("Failed to update query:", e))
    }
    setIsEditing(false)
  }

  const handleDelete = () => {
    setChatSessions(chatSessions.filter((s) => s.id !== session.id))
    
    // Delete from backend
    fetch(`/api/queries/delete?id=${session.id}`, {
      method: "DELETE",
      credentials: "include",
    }).catch(e => console.error("Failed to delete query:", e))
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      handleRename()
    } else if (e.key === "Escape") {
      setEditName(session.name)
      setIsEditing(false)
    }
  }

  if (isEditing) {
    return (
      <div className="flex items-center gap-1 pl-6 pr-2 py-1.5">
        <input
          ref={inputRef}
          type="text"
          value={editName}
          onChange={(e) => setEditName(e.target.value)}
          onKeyDown={handleKeyDown}
          onBlur={handleRename}
          className={cn(
            "flex-1 text-xs px-1.5 py-1 rounded border bg-transparent outline-none min-w-0",
            theme === "dark" ? "border-white/20 text-white" : "border-gray-300 text-gray-900"
          )}
        />
        <button onClick={handleRename} className={cn("p-1 rounded hover:bg-white/10", theme === "dark" ? "text-green-400" : "text-green-600")}>
          <Check className="h-3 w-3" />
        </button>
        <button onClick={() => { setEditName(session.name); setIsEditing(false) }} className={cn("p-1 rounded hover:bg-white/10", theme === "dark" ? "text-gray-400" : "text-gray-600")}>
          <X className="h-3 w-3" />
        </button>
      </div>
    )
  }

  return (
    <div className="group relative flex items-center">
      <Link
        href={`/playground/${session.id}`}
        className={cn(
          "flex items-center rounded-lg pl-6 pr-2 py-2 text-xs transition-colors flex-1 min-w-0",
          theme === "dark" ? "text-gray-500 hover:bg-white/5 hover:text-gray-300" : "text-gray-500 hover:bg-gray-100 hover:text-gray-700"
        )}
        title={session.name}
      >
        <span className="truncate">{session.name}</span>
      </Link>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <button className={cn("absolute right-1 p-1 rounded opacity-0 group-hover:opacity-100 transition-opacity", theme === "dark" ? "hover:bg-white/10 text-gray-400" : "hover:bg-gray-200 text-gray-600")}>
            <MoreHorizontal className="h-3.5 w-3.5" />
          </button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" className="w-32">
          <DropdownMenuItem onClick={() => setIsEditing(true)} className="gap-2 text-xs">
            <Pencil className="h-3 w-3" />
            Rename
          </DropdownMenuItem>
          <DropdownMenuItem onClick={handleDelete} className="gap-2 text-xs text-red-500 focus:text-red-500">
            <Trash2 className="h-3 w-3" />
            Delete
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    </div>
  )
}

export function Sidebar() {
  const pathname = usePathname()
  const router = useRouter()
  const { user, logout } = useAuth()
  const { collapsed, setCollapsed, theme, setTheme, chatSessions } = useSidebar()
  const [playgroundExpanded, setPlaygroundExpanded] = useState(true)
  const [helpModalOpen, setHelpModalOpen] = useState(false)
  const [bookingModalOpen, setBookingModalOpen] = useState(false)

  const toggleTheme = () => {
    setTheme(theme === "dark" ? "light" : "dark")
  }

  const isPlaygroundActive = pathname.startsWith("/playground")

  const handleNewChat = () => {
    router.push("/playground?new=" + Date.now())
  }

  const handleLogout = async () => {
    await logout()
  }

  const getUserInitials = () => {
    if (!user?.name) return "SL"
    const names = user.name.split(" ")
    if (names.length >= 2) {
      return names[0][0] + names[1][0]
    }
    return user.name.substring(0, 2).toUpperCase()
  }

  return (
    <aside
      className={cn(
        "fixed left-0 top-0 z-40 flex h-screen flex-col border-r transition-all duration-300",
        theme === "dark" ? "border-white/10 bg-[#0A0A0B]" : "border-gray-200 bg-white",
        collapsed ? "w-16" : "w-64"
      )}
    >
      <div className={cn("flex h-16 items-center gap-2 border-b px-4", theme === "dark" ? "border-white/10" : "border-gray-200")}>
        {!collapsed && (
          <>
            <Image src={theme === "dark" ? "/images/schema-light.png" : "/images/schema-dark.png"} alt="Schema" width={72} height={20} className="h-5 w-auto" priority />
            <span className="ml-auto rounded bg-[#0052CC]/20 px-1.5 py-0.5 font-mono text-[10px] text-[#2684FF]">ALPHA</span>
          </>
        )}
        {collapsed && <span className={cn("mx-auto text-lg font-semibold", theme === "dark" ? "text-white" : "text-gray-900")}>S</span>}
      </div>

      <nav className="flex-1 overflow-y-auto space-y-1 px-3 py-4">
        {navigation.map((item) => {
          const isActive = pathname === item.href || (item.href !== "/" && pathname.startsWith(item.href))
          return (
            <Link
              key={item.name}
              href={item.href}
              className={cn(
                "flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-colors",
                isActive ? "bg-[#0052CC]/15 text-[#2684FF]" : theme === "dark" ? "text-gray-400 hover:bg-white/5 hover:text-white" : "text-gray-600 hover:bg-gray-100 hover:text-gray-900",
                collapsed && "justify-center px-2"
              )}
              title={collapsed ? item.name : undefined}
            >
              <item.icon className="h-5 w-5 shrink-0" />
              {!collapsed && item.name}
            </Link>
          )
        })}

        <div className="pt-2">
          {collapsed ? (
            <Link href="/playground" className={cn("flex items-center justify-center rounded-lg px-2 py-2.5 text-sm font-medium transition-colors", isPlaygroundActive ? "bg-[#0052CC]/15 text-[#2684FF]" : theme === "dark" ? "text-gray-400 hover:bg-white/5 hover:text-white" : "text-gray-600 hover:bg-gray-100 hover:text-gray-900")} title="Playground">
              <MessageSquare className="h-5 w-5 shrink-0" />
            </Link>
          ) : (
            <>
              <button onClick={() => setPlaygroundExpanded(!playgroundExpanded)} className={cn("flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-colors", isPlaygroundActive ? "bg-[#0052CC]/15 text-[#2684FF]" : theme === "dark" ? "text-gray-400 hover:bg-white/5 hover:text-white" : "text-gray-600 hover:bg-gray-100 hover:text-gray-900")}>
                <MessageSquare className="h-5 w-5 shrink-0" />
                <span className="flex-1 text-left">Playground</span>
                {playgroundExpanded ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
              </button>
              {playgroundExpanded && (
                <div className={cn("ml-4 mt-1 space-y-0.5 border-l pl-3 max-h-64 overflow-y-auto [&::-webkit-scrollbar]:w-0 hover:[&::-webkit-scrollbar]:w-1.5 [&::-webkit-scrollbar-track]:bg-transparent [&::-webkit-scrollbar-thumb]:bg-muted-foreground/30 [&::-webkit-scrollbar-thumb]:rounded-full", theme === "dark" ? "border-white/10" : "border-gray-200")}>
                  <button onClick={handleNewChat} className={cn("flex items-center gap-2 rounded-lg px-2 py-2 text-xs font-medium transition-colors w-full text-left", theme === "dark" ? "text-[#2684FF] hover:bg-white/5" : "text-[#0052CC] hover:bg-gray-100")}>
                    <Plus className="h-3.5 w-3.5" />
                    New Chat
                  </button>
                  {chatSessions.length === 0 ? (
                    <p className={cn("pl-6 py-2 text-xs", theme === "dark" ? "text-gray-600" : "text-gray-400")}>No chats yet</p>
                  ) : (
                    <>
                      {chatSessions.map((session) => (
                        <ChatSessionItem key={session.id} session={session} theme={theme} />
                      ))}

                    </>
                  )}
                </div>
              )}
            </>
          )}
        </div>

        <Link href="/configuration" className={cn("flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-colors", pathname === "/configuration" ? "bg-[#0052CC]/15 text-[#2684FF]" : theme === "dark" ? "text-gray-400 hover:bg-white/5 hover:text-white" : "text-gray-600 hover:bg-gray-100 hover:text-gray-900", collapsed && "justify-center px-2")} title={collapsed ? "Configuration" : undefined}>
          <Settings className="h-5 w-5 shrink-0" />
          {!collapsed && "Configuration"}
        </Link>
      </nav>

      <div className={cn("border-t px-3 py-4 space-y-1", theme === "dark" ? "border-white/10" : "border-gray-200")}>
        <button onClick={() => setCollapsed(!collapsed)} className={cn("flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-colors w-full", theme === "dark" ? "text-gray-400 hover:bg-white/5 hover:text-white" : "text-gray-600 hover:bg-gray-100 hover:text-gray-900", collapsed && "justify-center px-2")} title={collapsed ? "Expand" : "Collapse"}>
          {collapsed ? <PanelLeft className="h-5 w-5 shrink-0" /> : <><PanelLeftClose className="h-5 w-5 shrink-0" /><span>Collapse</span></>}
        </button>
        <button onClick={() => setHelpModalOpen(true)} className={cn("flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-colors w-full", theme === "dark" ? "text-gray-400 hover:bg-white/5 hover:text-white" : "text-gray-600 hover:bg-gray-100 hover:text-gray-900", collapsed && "justify-center px-2")} title={collapsed ? "Help" : undefined}>
          <HelpCircle className="h-5 w-5 shrink-0" />
          {!collapsed && "Help"}
        </button>
        <button onClick={toggleTheme} className={cn("flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-colors w-full", theme === "dark" ? "text-gray-400 hover:bg-white/5 hover:text-white" : "text-gray-600 hover:bg-gray-100 hover:text-gray-900", collapsed && "justify-center px-2")} title={theme === "dark" ? "Switch to Light Mode" : "Switch to Dark Mode"}>
          {theme === "dark" ? <><Sun className="h-5 w-5 shrink-0" />{!collapsed && <span>Light Mode</span>}</> : <><Moon className="h-5 w-5 shrink-0" />{!collapsed && <span>Dark Mode</span>}</>}
        </button>
      </div>

      {!collapsed && (
        <div className={cn("border-t p-4", theme === "dark" ? "border-white/10" : "border-gray-200")}>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <button className="flex items-center gap-3 w-full rounded-lg p-1 -m-1 transition-colors hover:bg-white/5">
                <div className="flex h-9 w-9 items-center justify-center rounded-full bg-gradient-to-br from-[#2684FF] to-[#0052CC] text-sm font-medium text-white shrink-0">
                  {getUserInitials()}
                </div>
                <div className="flex-1 text-left min-w-0">
                  <p className={cn("text-sm font-medium truncate", theme === "dark" ? "text-white" : "text-gray-900")}>{user?.name || "User"}</p>
                  <p className={cn("text-xs truncate", theme === "dark" ? "text-gray-500" : "text-gray-500")}>{user?.email || "user@schemalabs.ai"}</p>
                </div>
              </button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="start" side="top" className="w-56 mb-2">
              <Link href="/account"><DropdownMenuItem className="gap-3 cursor-pointer"><User className="h-4 w-4" />Account</DropdownMenuItem></Link>
              <Link href="/billing"><DropdownMenuItem className="gap-3 cursor-pointer"><CreditCard className="h-4 w-4" />Billing</DropdownMenuItem></Link>
              <Link href="/usage"><DropdownMenuItem className="gap-3 cursor-pointer"><BarChart3 className="h-4 w-4" />Usage</DropdownMenuItem></Link>
              <DropdownMenuItem onClick={handleLogout} className="gap-3 cursor-pointer text-red-500 focus:text-red-500"><LogOut className="h-4 w-4" />Log out</DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      )}
      {collapsed && (
        <div className={cn("border-t p-3 flex justify-center", theme === "dark" ? "border-white/10" : "border-gray-200")}>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <button className="flex h-9 w-9 items-center justify-center rounded-full bg-gradient-to-br from-[#2684FF] to-[#0052CC] text-sm font-medium text-white transition-opacity hover:opacity-80">
                {getUserInitials()}
              </button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="center" side="top" className="w-56 mb-2">
              <div className="px-2 py-1.5 text-xs text-muted-foreground">{user?.email || "user@schemalabs.ai"}</div>
              <Link href="/account"><DropdownMenuItem className="gap-3 cursor-pointer"><User className="h-4 w-4" />Account</DropdownMenuItem></Link>
              <Link href="/billing"><DropdownMenuItem className="gap-3 cursor-pointer"><CreditCard className="h-4 w-4" />Billing</DropdownMenuItem></Link>
              <Link href="/usage"><DropdownMenuItem className="gap-3 cursor-pointer"><BarChart3 className="h-4 w-4" />Usage</DropdownMenuItem></Link>
              <DropdownMenuItem onClick={handleLogout} className="gap-3 cursor-pointer text-red-500 focus:text-red-500"><LogOut className="h-4 w-4" />Log out</DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      )}

      <Dialog open={helpModalOpen} onOpenChange={setHelpModalOpen}>
        <DialogContent className="border-border bg-card sm:max-w-[400px]">
          <DialogHeader>
            <DialogTitle className="text-foreground">How can we help?</DialogTitle>
            <DialogDescription className="text-muted-foreground">Choose an option below to get support</DialogDescription>
          </DialogHeader>
          <div className="grid gap-3 py-4">
            <Button variant="outline" className="h-auto p-4 justify-start gap-4 bg-transparent hover:bg-muted/50" onClick={() => { setHelpModalOpen(false); setBookingModalOpen(true) }}>
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-[#0052CC]/10"><Calendar className="h-5 w-5 text-[#0052CC] dark:text-[#2684FF]" /></div>
              <div className="text-left"><p className="font-medium text-foreground">Book a Meeting</p><p className="text-xs text-muted-foreground">Schedule a call with our team</p></div>
            </Button>
            <Button variant="outline" className="h-auto p-4 justify-start gap-4 bg-transparent hover:bg-muted/50" onClick={() => { window.location.href = "mailto:support@schemalabs.ai?subject=Support%20Request"; setHelpModalOpen(false) }}>
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-[#0052CC]/10"><Mail className="h-5 w-5 text-[#0052CC] dark:text-[#2684FF]" /></div>
              <div className="text-left"><p className="font-medium text-foreground">Support</p><p className="text-xs text-muted-foreground">Email us at support@schemalabs.ai</p></div>
            </Button>
          </div>
        </DialogContent>
      </Dialog>

      <Dialog open={bookingModalOpen} onOpenChange={setBookingModalOpen}>
        <DialogContent className="border-border bg-card sm:max-w-[600px] max-h-[80vh]">
          <DialogHeader>
            <DialogTitle className="text-foreground">Book a Meeting</DialogTitle>
            <DialogDescription className="text-muted-foreground">Schedule a call with our team</DialogDescription>
          </DialogHeader>
          <div className="py-4">
            <div className="flex flex-col items-center justify-center h-[400px] rounded-lg border border-dashed border-border bg-muted/30">
              <Calendar className="h-12 w-12 text-muted-foreground mb-4" />
              <p className="text-sm text-muted-foreground text-center">Google Calendar booking will be embedded here</p>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </aside>
  )
}
