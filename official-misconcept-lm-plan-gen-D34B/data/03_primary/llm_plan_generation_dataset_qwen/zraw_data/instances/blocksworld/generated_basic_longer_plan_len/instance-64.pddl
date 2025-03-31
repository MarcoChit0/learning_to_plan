

(define (problem BW-rand-7)
(:domain blocksworld-4ops)
(:objects a b c d e f g )
(:init
(handempty)
(on a g)
(on b f)
(on c e)
(ontable d)
(on e d)
(ontable f)
(on g b)
(clear a)
(clear c)
)
(:goal
(and
(on a f)
(on b c)
(on c a)
(on d g)
(on e d)
(on f e))
)
)


