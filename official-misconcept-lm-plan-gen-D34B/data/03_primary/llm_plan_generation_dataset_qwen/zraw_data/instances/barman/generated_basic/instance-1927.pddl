(define (problem prob)
 (:domain barman)
 (:objects 
      shaker357 - shaker
      left right - hand
      shot198 - shot
      ingredient443 ingredient442 - ingredient
      cocktail296 - cocktail
      dispenser291 dispenser79 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker357)
  (ontable shot198)
  (dispenses dispenser291 ingredient443)
  (dispenses dispenser79 ingredient442)
  (clean shaker357)
  (clean shot198)
  (empty shaker357)
  (empty shot198)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker357 l0)
  (shaker-level shaker357 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail296 ingredient442)
  (cocktail-part2 cocktail296 ingredient443)
)
 (:goal
  (and
      (contains shot198 cocktail296)
)))
