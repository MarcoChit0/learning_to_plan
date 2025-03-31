

(define (problem BW-rand-7)
(:domain blocksworld-4ops)
(:objects a b c d e f g )
(:init
(handempty)
(on a f)
(on b a)
(on c d)
(on d g)
(on e c)
(ontable f)
(on g b)
(clear e)
)
(:goal
(and
(on a c)
(on b a)
(on d e)
(on e b)
(on f d)
(on g f))
)
)


