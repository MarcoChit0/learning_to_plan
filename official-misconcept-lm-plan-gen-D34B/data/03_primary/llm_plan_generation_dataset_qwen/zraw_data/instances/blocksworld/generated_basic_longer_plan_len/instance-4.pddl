

(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects a b c d e f )
(:init
(handempty)
(on a e)
(ontable b)
(on c a)
(on d c)
(on e b)
(on f d)
(clear f)
)
(:goal
(and
(on a e)
(on b f)
(on c b)
(on d c))
)
)


