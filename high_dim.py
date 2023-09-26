import string
import typing
import numpy
import cirq

"""
# Set precision of numpy complex value.
# numpy.complex256 = float128*real + float128*imag
"""

complex_ = numpy.complex256
img =complex_(0+1j)
one =complex_(1+0j)

def generate_weyl_operators(p: int, q: int, root=1, dim: int=2, rotation: float=1.0, adjoint: bool = False) -> numpy.ndarray:
    """
    Generates the weyl transformation matrixes.
    
    Arg:
        q: int: 0 to dim integer
        p: int: 0 to dim integer
        root: int: Root the transformation.
        dim: int: Dimension of Weyl.
        rotation: float: Rotation of transformation.
        adjoint: bool: Generate hermitian.
    
    Returns:
        matrix: numpy.ndarray
    """
    
    if not dim > p >= 0 or not dim > q >= 0:
        raise ValueError(f'q and p must be in between 0 and {dim}!')        
    
    ##------------------------------------------------
    ## Important !!
    ## w = e^^(2*Pi*rot*i)/(d^^k)
    ##------------------------------------------------
    
    w = numpy.exp(((2*numpy.pi*rotation)/(numpy.power(dim, root)))*img)
    matrix = numpy.zeros((dim, dim), dtype=complex_)
    
    for k in range(dim):
        if not adjoint:
            amp = numpy.power(w, k*p)
            s = numpy.zeros((dim, 1), dtype=complex_)
            s[k] += one
            m = numpy.zeros((dim, 1), dtype=complex_)
            m[(k+q)%dim] += one
            m = numpy.transpose(numpy.conjugate(m))
        else:
            amp = numpy.power(w, k*q)
            s = numpy.zeros((dim, 1), dtype=complex_)
            s[(k+p)%dim] += one
            m = numpy.zeros((dim, 1), dtype=complex_)
            m[k] += one
            m = numpy.transpose(numpy.conjugate(m))
            
        matrix += amp * numpy.dot(s, m)
    return matrix

def identity(dim: int = 2) -> numpy.ndarray:
    """
    Generates dimesion identity matrix.
    
    Args:
        dim: int: Dimension.
        
    Return:
        matrix: numpy.ndarray
    """
    
    return generate_weyl_operators(0, 0, dim=dim, adjoint=True)

def general_X(p: int, dim: int = 2) -> numpy.ndarray:
    """
    Generates dimesion X matrix.
    
    Args:
        p: int: Version of X.
        dim: int: Dimension.
        
    Return:
        matrix: numpy.ndarray
    """
    
    return generate_weyl_operators(p, 0, dim=dim, adjoint=True)

def general_Y(p: int, dim: int = 2) -> numpy.ndarray:
    """
    Generates dimesion Y matrix.
    
    Args:
        p: int: Version of Y.
        dim: int: Dimension.
        
    Return:
        matrix: numpy.ndarray
    """
    
    gp = numpy.power(img, p%dim) 
    return gp*generate_weyl_operators(p, p, dim=dim, adjoint=True)

def general_Z(q: int, root: int=1, rotation: float=1.0, dim: int = 2) -> numpy.ndarray:
    """
    Generates dimesion Z matrix.
    
    Args:
        q: int: Version of Z.
        dim: int: Dimension.
        root: int: Root the transformation.
        rotation: float: Rotation of transformation.
        
    Return:
        matrix: numpy.ndarray
    """
    
    return generate_weyl_operators(0, q, root=root, rotation=rotation, dim=dim, adjoint=True)

def general_CX(dim: int = 2) -> numpy.ndarray:
    """
    Generates dimesion CX matrix.
    Multi-Value (1, 2 .... dimension-1)  ---> X.
    
    Args:
        dim: int: Dimension.
        
    Return:
        matrix: numpy.ndarray
    """
        
    matrix = numpy.zeros((numpy.power(dim, 2), numpy.power(dim, 2)), dtype=complex_)
    
    for k in range(dim):
        s = numpy.zeros((dim, 1), dtype=complex_)
        s[k] += one
        pk = numpy.dot(s, numpy.transpose(numpy.conjugate(s)))
        pk = numpy.kron(pk, general_X(k, dim=dim))
        matrix += pk
    return matrix

def general_CY(dim: int = 2) -> numpy.ndarray:
    """
    Generates dimesion CY matrix.
    Multi-Value (1, 2 .... dimension-1)  ---> Y.
    
    Args:
        dim: int: Dimension.
        
    Return:
        matrix: numpy.ndarray
    """
        
    matrix = numpy.zeros((numpy.power(dim, 2), numpy.power(dim, 2)), dtype=complex_)
    
    for k in range(dim):
        s = numpy.zeros((dim, 1), dtype=complex_)
        s[k] +=complex_(1 + 0j)
        pk = numpy.dot(s, numpy.transpose(numpy.conjugate(s)))
        pk = numpy.kron(pk, general_Y(k, dim=dim))
        matrix += pk
    return matrix

def general_CZ(root: int=1, rotation: float=1.0, dim: int = 2) -> numpy.ndarray:
    """
    Generates dimesion CZ matrix.
    Multi-Value (1, 2 .... dimension-1)  ---> Z
    
    Args:
        root: int: Root the transformation.
        rotation: float: Rotation of transformation.
        dim: int: Dimension.
        
    Return:
        matrix: numpy.ndarray
    """
        
    matrix = numpy.zeros((numpy.power(dim, 2), numpy.power(dim, 2)), dtype=complex_)
    
    for k in range(dim):
        s = numpy.zeros((dim, 1), dtype=complex_)
        s[k] += one
        pk = numpy.dot(s, numpy.transpose(numpy.conjugate(s)))
        pk = numpy.kron(pk, general_Z(k, root=root, rotation=rotation, dim=dim))
        matrix += pk
    return matrix

def general_QFT(n: int, dim: int=2) -> numpy.ndarray:
    """
    Generates the general Quantum Fourier Transformation.
    Warning: It is slow right now.
    
    Args:
        n: int: Qudit count.
        dim: int: Dimension.
    
    Return:
        matrix: numpy.ndarray
    """
    
    N = numpy.power(dim, n)
    matrix = numpy.full((N, N), fill_value=numpy.exp((2*numpy.pi/N)*img), dtype=complex_)
    
    m, n = matrix.shape
    x, y = numpy.meshgrid(numpy.arange(m), numpy.arange(n), indexing='ij')
    return 1 / numpy.sqrt(N) * numpy.power(matrix, x*y) 

def general_Hadamard(dim: int=2) -> numpy.ndarray:
    """
    Generates the general Hadamard Transformation.
    
    Args:
        dim: int: Dimension.
    
    Return:
        matrix: numpy.ndarray
    """
    
    return general_QFT(1, dim=dim)

def general_Hadamard_Adjoint(dim: int=2) -> numpy.ndarray:
    """
    Generates the general Hadamard Transformation.
    
    Args:
        dim: int: Dimension.
    
    Return:
        matrix: numpy.ndarray
    """
    
    return numpy.transpose(numpy.conjugate(general_QFT(1, dim=dim)))

def swap(dim: int=2) -> numpy.ndarray:
    """
    Generates the SWAP gate matrix.
    
    Args:
        dim: int: Dimension.
        
    Returns:
        matrix: numpy.ndarray.
    """
    
    d_2 = numpy.power(dim, 2)
    matrix = numpy.zeros((d_2, d_2), dtype=complex_)
    
    k, m = 0, 0
    for i in range(dim):
        for j in range(dim):
            matrix[k][j*dim + m] += one 
            k += 1
        m += 1
    return matrix

def find_eigenphases(dim) -> typing.Dict[int,complex_]:
    """
    Finds the eigenphases of the given dimension.
    
    Args:
        dim: int: Dimension.
        
    Returns:
        dictionary: typing.Dict[int,complex_]
    """
    
    wyo = generate_weyl_operators(1, 0, root=1, dim=dim, adjoint=False)
    ep_dict: typing.Dict[int,complex_] = dict()
    
    for i in range(dim):
        ep_dict[i] = wyo[i][i]
    return ep_dict

digs = string.digits + string.ascii_letters

def int2base(x: int, base: int, padding: int=0, big_endian: bool=False) -> str:
    """
    Convert X_{base} value to X_{10}.
    
    Args:
        x: int: Value.
        base: int: Base of value.
        padding: int: Pad 0's before.
        big_endian: bool: Set encoding.
    
    Return:
        rebased value: str.
    """
    
    if x < 0:
        sign = -1
    elif x == 0:
        return digs[0]*padding
    else:
        sign = 1

    x *= sign
    digits = []

    while x:
        digits.append(digs[x % base])
        x = x // base
    
    if digits.__len__() < padding:
        digits.append('0'*(padding-digits.__len__()))
        
    if sign < 0:
        digits.append('-')
    
    if not big_endian:
        digits.reverse()
    return ''.join(digits)

def get_histogram(resaults: cirq.Result, key: str='M', padding: bool=False) -> typing.Dict:
    """
    Creates the histogram of Cirq results.
    
    Args:
        resault: cirq.Results
        key: str: key of measurement.
        padding: bool: padd all possibilities.
    
    Return:
        histogram: typing.Dict.
    """
    
    meas = {}
    
    if padding:
        for i in range(numpy.power(d, n)):
            meas[int2base(i, d, padding=n)] = 0

    for i in range(resaults.measurements[key].__len__()):
        s = ''
        for j in range(resaults.measurements[key][i].__len__()):
            s += str(resaults.measurements[key][i][j])
        try:
            meas[s] += 1
        except KeyError:
            meas[s] = 1
    
    keys = list(meas.keys())
    keys.sort()
    meas = {i: meas[i] for i in keys}
    
    return meas

class Gate(cirq.Gate):
    def __init__(self, unitary_matrix: typing.Iterable, dimension: int, 
                 diag_info: typing.Union[cirq.CircuitDiagramInfo, typing.Tuple[str], str, None] = None):
        """
        Cirq create a gate from unitary matrix in arbitary d-Dimension.
        
        Args:
            unitary_matrix: Sequance: Transform matrix.
            dimension: int: Dimension.
            diag_info: cirq.CircuitDiagramInfo, Tuple[str], str, None: Diagram to show.
            
        Raises:
            Exception: If not a quantum transformation matrix. 
        """
        
        if not isinstance(unitary_matrix, numpy.ndarray):
            try:
                unitary_matrix = numpy.array(unitary_matrix, dtype=complex_)
                unitary_matrix.reshape(unitary_matrix.__len__(), unitary_matrix.__len__())
            except Exception:
                raise Exception(f'The matrix to numpy matrix conversion is failed!')
        
        sh = unitary_matrix.shape
        shl = sh.__len__()
        if shl > 2:
            raise ValueError(f'The matrix needs to be a 2D. {2} != {shl}')
            
        if sh[0] != sh [1]:
            raise ValueError(f'The matrix shapes need to match. {sh[0]} != {sh[1]}')
        
        if not isinstance(dimension, int) or dimension <= 1:
            raise ValueError(f'The dimension needs to be an intager that equal or bigger then 2. {dimension}?')
        
        qcount = numpy.emath.logn(dimension, sh[0])
        qcount = numpy.round(qcount, 4)
        
        if not qcount.is_integer():
            raise AttributeError(f'The dimension and traform matrix do not match. log_{dimension}^{sh[0]} is not integer!') 
            
        self.unitary_matrix = unitary_matrix
        self.dimension = dimension
        self.qubit_count = int(qcount)
        
        if diag_info is None:
            diag_info = cirq.CircuitDiagramInfo(wire_symbols=('U', ) * self.qubit_count)
            
        elif isinstance(diag_info, str):
            diag_info = cirq.CircuitDiagramInfo(wire_symbols=(diag_info, ) * self.qubit_count)
            
        elif isinstance(diag_info, typing.Tuple):
            diag_info = cirq.CircuitDiagramInfo(wire_symbols=diag_info)
        
        elif isinstance(diag_info, cirq.CircuitDiagramInfo):
            pass
        
        else:
            raise ValueError(f'Diagram info must be a circuit diagram info class.')
            
        self.gate_name = diag_info
        super().__init__()
        
    def _qid_shape_(self) -> typing.Tuple[int]:
        return (self.dimension, ) * self.qubit_count
    
    def _num_qubits_(self) -> int:
        return self.qubit_count

    def _unitary_(self) -> numpy.ndarray:
        return self.unitary_matrix

    def _circuit_diagram_info_(self, args) -> cirq.CircuitDiagramInfo:
        return self.gate_name
    
    def __str__(self) -> str:
        return f'{self.unitary_matrix}'
    
    @property
    def transform_matrix(self) -> numpy.ndarray:
        return self.unitary_matrix