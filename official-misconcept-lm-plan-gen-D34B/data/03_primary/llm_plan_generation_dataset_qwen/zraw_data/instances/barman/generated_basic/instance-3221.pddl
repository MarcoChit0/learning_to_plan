(define (problem prob)
 (:domain barman)
 (:objects 
      shaker356 - shaker
      left right - hand
      shot123 - shot
      ingredient260 ingredient365 - ingredient
      cocktail352 - cocktail
      dispenser127 dispenser258 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker356)
  (ontable shot123)
  (dispenses dispenser127 ingredient260)
  (dispenses dispenser258 ingredient365)
  (clean shaker356)
  (clean shot123)
  (empty shaker356)
  (empty shot123)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker356 l0)
  (shaker-level shaker356 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail352 ingredient260)
  (cocktail-part2 cocktail352 ingredient365)
)
 (:goal
  (and
      (contains shot123 cocktail352)
)))
