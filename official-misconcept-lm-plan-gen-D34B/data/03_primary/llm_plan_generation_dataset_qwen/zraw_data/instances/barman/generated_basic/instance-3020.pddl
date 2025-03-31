(define (problem prob)
 (:domain barman)
 (:objects 
      shaker305 - shaker
      left right - hand
      shot237 - shot
      ingredient497 ingredient379 ingredient177 - ingredient
      cocktail323 - cocktail
      dispenser146 dispenser312 dispenser257 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker305)
  (ontable shot237)
  (dispenses dispenser146 ingredient497)
  (dispenses dispenser312 ingredient379)
  (dispenses dispenser257 ingredient177)
  (clean shaker305)
  (clean shot237)
  (empty shaker305)
  (empty shot237)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker305 l0)
  (shaker-level shaker305 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail323 ingredient379)
  (cocktail-part2 cocktail323 ingredient177)
)
 (:goal
  (and
      (contains shot237 cocktail323)
)))
