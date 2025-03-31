(define (problem prob)
 (:domain barman)
 (:objects 
      shaker122 - shaker
      left right - hand
      shot390 - shot
      ingredient248 ingredient56 - ingredient
      cocktail223 - cocktail
      dispenser14 dispenser244 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker122)
  (ontable shot390)
  (dispenses dispenser14 ingredient248)
  (dispenses dispenser244 ingredient56)
  (clean shaker122)
  (clean shot390)
  (empty shaker122)
  (empty shot390)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker122 l0)
  (shaker-level shaker122 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail223 ingredient248)
  (cocktail-part2 cocktail223 ingredient56)
)
 (:goal
  (and
      (contains shot390 cocktail223)
)))
