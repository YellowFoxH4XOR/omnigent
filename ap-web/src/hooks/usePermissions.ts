/**
 * TanStack Query hooks for session permissions CRUD.
 * Wraps the fetch functions in `permissionsApi.ts`.
 */

import { useMemo } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  type Permission,
  derivePermissionLevel,
  grantPermission,
  listPermissions,
  revokePermission,
} from "@/lib/permissionsApi";
import { useConversations } from "./useConversations";
import { useSession } from "./useSession";

function permissionsKey(sessionId: string) {
  return ["permissions", sessionId] as const;
}

/** Fetch all permission grants for a session. */
export function usePermissions(sessionId: string | null) {
  return useQuery({
    queryKey: permissionsKey(sessionId ?? ""),
    queryFn: () => listPermissions(sessionId!),
    enabled: !!sessionId,
  });
}

/** Grant or update a permission. Invalidates the permissions list on success. */
export function useGrantPermission(sessionId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ userId, level }: { userId: string; level: number }) =>
      grantPermission(sessionId, userId, level),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: permissionsKey(sessionId) });
    },
  });
}

/** Revoke a permission. Invalidates the permissions list on success. */
export function useRevokePermission(sessionId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (userId: string) => revokePermission(sessionId, userId),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: permissionsKey(sessionId) });
    },
  });
}

/**
 * Returns whether the current user has edit access (level >= 2) to a session.
 * `null` permission level (single-user mode) is treated as unrestricted.
 */
export function useCanEdit(conversationId: string): boolean {
  const { data: conversationsData } = useConversations();
  const { session: activeSession, isLoading: sessionLoading } = useSession(conversationId);
  return useMemo(() => {
    const conversations = conversationsData?.pages.flatMap((p) => p.data);
    const activeConv = conversations?.find((c) => c.id === conversationId) ?? null;
    const permissionLevel = derivePermissionLevel(
      activeSession,
      sessionLoading,
      activeConv,
      conversationId,
      conversationsData !== undefined,
    );
    return permissionLevel == null || permissionLevel >= 2;
  }, [conversationsData, conversationId, activeSession, sessionLoading]);
}

export type { Permission };
