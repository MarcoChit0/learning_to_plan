(define (problem prob)
 (:domain barman)
 (:objects 
      shaker293 - shaker
      left right - hand
      shot19 - shot
      ingredient130 ingredient160 - ingredient
      cocktail435 - cocktail
      dispenser229 dispenser286 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker293)
  (ontable shot19)
  (dispenses dispenser229 ingredient130)
  (dispenses dispenser286 ingredient160)
  (clean shaker293)
  (clean shot19)
  (empty shaker293)
  (empty shot19)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker293 l0)
  (shaker-level shaker293 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail435 ingredient160)
  (cocktail-part2 cocktail435 ingredient130)
)
 (:goal
  (and
      (contains shot19 cocktail435)
)))
