

(define (problem BW-rand-7)
(:domain blocksworld-4ops)
(:objects a b c d e f g )
(:init
(handempty)
(on a e)
(on b d)
(on c f)
(on d c)
(ontable e)
(on f g)
(ontable g)
(clear a)
(clear b)
)
(:goal
(and
(on a g)
(on b f)
(on c a)
(on d c)
(on e b)
(on g e))
)
)


