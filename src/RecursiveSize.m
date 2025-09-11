function bytes = RecursiveSize(v, visited)
%RECURSIVESIZE  Deep (recursive) size estimate, with handle de-duping.
%
%   BYTES = RECURSIVESIZE(V)  returns an *approximate* number of bytes
%   occupied by V **and everything reachable from it** (cells, structs,
%   object properties…).  Handle objects are counted only once.
%
%   Limitations: copy-on-write sharing and MATLAB’s per-object header
%   remain invisible to pure-MATLAB code, so numbers are still estimates.
%
%   © 2025  (BSD-style licence – free to use with attribution)

%--------------------------------------------------------------------------
% 0.  initial call – create map of visited handle IDs
%--------------------------------------------------------------------------
if nargin == 1
    visited = containers.Map('KeyType','char','ValueType','logical');
end

%--------------------------------------------------------------------------
% 1.  deduplicate HANDLE objects (graphics, Java, user handle classes)
%--------------------------------------------------------------------------
if ishandle(v)
    key = sprintf('%s_%d', class(v), double(v));   % stable identifier
    if isKey(visited, key)
        bytes = 0;   % already counted – bail out
        return
    end
    visited(key) = true;
end

%--------------------------------------------------------------------------
% 2.  primitive / non-container values – just ask WHOS
%--------------------------------------------------------------------------
if ~iscell(v) && ~isstruct(v) && ~isobject(v)
    info  = whos('v');      % local symbol literally named 'v'
    bytes = info.bytes;
    return
end

%--------------------------------------------------------------------------
% 3.  container types – recurse
%--------------------------------------------------------------------------
bytes = 0;

%––– 3a. cells ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
if iscell(v)
    for k = 1:numel(v)
        bytes = bytes + RecursiveSize(v{k}, visited);
    end
    return
end

%––– 3b. structs ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
if isstruct(v)
    fn = fieldnames(v);
    for k = 1:numel(fn)
        fld = {v.(fn{k})};
        for j = 1:numel(fld)
            bytes = bytes + RecursiveSize(fld{j}, visited);
        end
    end
    return
end

%––– 3c. objects (value *or* handle) ––––––––––––––––––––––––––––––––––––––
if isobject(v)

    % Handle *arrays* of objects first ------------------------------------
    if numel(v) > 1
        for k = 1:numel(v)
            bytes = bytes + RecursiveSize(v(k), visited);
        end
        return
    end

    % Grab metadata – guard against exotic objects where METACLASS fails
    try
        mc = metaclass(v);
    catch
        % Fallback: treat like primitive
        info  = whos('v');
        bytes = info.bytes;
        return
    end

    if isempty(mc)           % metaclass might be [] for Java objects
        info  = whos('v');
        bytes = info.bytes;
        return
    end

    % mc can still be an *array* in rare cases; collapse it safely
    plist = [mc.PropertyList];       % horizontal concat avoids RHS error

    % Flags – flip to FALSE if you want them included
    skipHidden    = true;
    skipTransient = true;

    % Loop over each property
    for p = 1:numel(plist)
        prop = plist(p);

        if prop.Dependent
            continue    % avoids side-effect code
        end
        if skipHidden    && prop.Hidden
            continue
        end
        if skipTransient && prop.Transient
            continue
        end

        % Access the property – ignore any that error out
        try
            val   = v.(prop.Name);
            bytes = bytes + RecursiveSize(val, visited);
        catch   %#ok<CTCH>
        end
    end
end
