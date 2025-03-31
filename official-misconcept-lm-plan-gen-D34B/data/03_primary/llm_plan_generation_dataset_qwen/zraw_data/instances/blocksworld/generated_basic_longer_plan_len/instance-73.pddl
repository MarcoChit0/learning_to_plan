

(define (problem BW-rand-7)
(:domain blocksworld-4ops)
(:objects a b c d e f g )
(:init
(handempty)
(on a f)
(on b a)
(on c b)
(on d c)
(on e d)
(on f g)
(ontable g)
(clear e)
)
(:goal
(and
(on a g)
(on c b)
(on d f)
(on e c)
(on f a)
(on g e))
)
)


