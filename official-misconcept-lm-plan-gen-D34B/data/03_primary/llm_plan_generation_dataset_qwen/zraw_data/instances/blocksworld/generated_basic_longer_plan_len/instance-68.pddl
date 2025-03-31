

(define (problem BW-rand-7)
(:domain blocksworld-4ops)
(:objects a b c d e f g )
(:init
(handempty)
(on a g)
(ontable b)
(on c d)
(on d a)
(on e c)
(ontable f)
(on g b)
(clear e)
(clear f)
)
(:goal
(and
(on a c)
(on b d)
(on c e)
(on e f)
(on f b)
(on g a))
)
)


