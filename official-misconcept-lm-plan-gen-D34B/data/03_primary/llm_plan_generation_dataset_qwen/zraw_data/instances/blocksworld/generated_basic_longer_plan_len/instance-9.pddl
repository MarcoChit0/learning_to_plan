

(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects a b c d e f )
(:init
(handempty)
(on a d)
(on b f)
(on c a)
(on d b)
(on e c)
(ontable f)
(clear e)
)
(:goal
(and
(on a f)
(on b c)
(on c a)
(on f e))
)
)


