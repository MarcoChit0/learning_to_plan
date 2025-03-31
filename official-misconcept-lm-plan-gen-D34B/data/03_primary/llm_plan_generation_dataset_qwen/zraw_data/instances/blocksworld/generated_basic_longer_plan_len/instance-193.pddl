

(define (problem BW-rand-10)
(:domain blocksworld-4ops)
(:objects a b c d e f g h i j )
(:init
(handempty)
(on a d)
(ontable b)
(on c a)
(on d h)
(on e c)
(ontable f)
(on g f)
(on h b)
(on i e)
(ontable j)
(clear g)
(clear i)
(clear j)
)
(:goal
(and
(on a g)
(on b i)
(on c e)
(on d c)
(on e b)
(on f j)
(on g f)
(on j h))
)
)


