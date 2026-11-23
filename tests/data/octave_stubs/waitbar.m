function varargout = waitbar(varargin)
  % headless stand-in for waitbar, which mSOUND calls on every plane
  if nargout
    varargout{1} = [];
  end
end
