

(define (problem BW-rand-7)
(:domain blocksworld-4ops)
(:objects a b c d e f g )
(:init
(handempty)
(on a g)
(on b d)
(on c f)
(on d c)
(on e a)
(ontable f)
(on g b)
(clear e)
)
(:goal
(and
(on a e)
(on b d)
(on c a)
(on e f)
(on f b))
)
)


