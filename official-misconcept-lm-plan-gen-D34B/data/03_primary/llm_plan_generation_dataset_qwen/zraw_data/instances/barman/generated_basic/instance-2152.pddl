(define (problem prob)
 (:domain barman)
 (:objects 
      shaker402 - shaker
      left right - hand
      shot22 - shot
      ingredient68 ingredient299 - ingredient
      cocktail397 - cocktail
      dispenser380 dispenser149 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker402)
  (ontable shot22)
  (dispenses dispenser380 ingredient68)
  (dispenses dispenser149 ingredient299)
  (clean shaker402)
  (clean shot22)
  (empty shaker402)
  (empty shot22)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker402 l0)
  (shaker-level shaker402 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail397 ingredient68)
  (cocktail-part2 cocktail397 ingredient299)
)
 (:goal
  (and
      (contains shot22 cocktail397)
)))
