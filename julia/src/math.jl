@doc raw"""
    multidot(mats...)

Fast multiplication of multiple matrices, i.e. optimal order. Provide matrices in order of multiplication.
"""
function multidot(mats...)
    sizes = [size(mat, 1) for mat in mats]
    push!(sizes, size(mats[end], 2))
    order = matrix_chain_order(sizes)

    function optmul(i, j)
        if i == j
            mats[i]
        else
            optmul(i, order[i, j]) * optmul(order[i, j] + 1, j)
        end
    end

    return optmul(1, length(mats))
end

@doc raw"""
    matrix_chain_order(p)

Reference: Cormen, "Introduction to Algorithms", Chapter 15.2, p. 370-378
"""
function matrix_chain_order(p)
    n = length(p) - 1
    m = zeros(Int, n, n)
    s = zeros(Int, n, n)

    # subsequence lengths
    for l = 2:n
        for i = 1:(n - l + 1)
            j = i + l - 1
            m[i, j] = typemax(Int)
            for k = i:(j - 1)
                q = m[i, k] + m[k+1, j] + p[i]*p[k+1]*p[j+1]
                if q < m[i, j]
                    m[i, j] = q
                    s[i, j] = k
                end
            end
        end
    end

    return s
end

@doc raw"""
    optimal_print(s, i, j)

Helper function.
"""
function optimal_print(s, i, j)
    if i == j
        print("A[$i]")
    else
        print("(")
        optimal_print(s, i, s[i, j])
        print("*")
        optimal_print(s, s[i, j] + 1, j)
        print(")")
    end
end