(define (problem prob)
 (:domain barman)
 (:objects 
      shaker101 - shaker
      left right - hand
      shot150 - shot
      ingredient447 ingredient58 ingredient472 ingredient391 - ingredient
      cocktail194 - cocktail
      dispenser312 dispenser362 dispenser65 dispenser183 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker101)
  (ontable shot150)
  (dispenses dispenser312 ingredient447)
  (dispenses dispenser362 ingredient58)
  (dispenses dispenser65 ingredient472)
  (dispenses dispenser183 ingredient391)
  (clean shaker101)
  (clean shot150)
  (empty shaker101)
  (empty shot150)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker101 l0)
  (shaker-level shaker101 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail194 ingredient472)
  (cocktail-part2 cocktail194 ingredient447)
)
 (:goal
  (and
      (contains shot150 cocktail194)
)))
