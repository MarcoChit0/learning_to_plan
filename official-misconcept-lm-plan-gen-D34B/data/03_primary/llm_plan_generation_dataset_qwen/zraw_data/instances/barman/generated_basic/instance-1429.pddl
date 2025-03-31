(define (problem prob)
 (:domain barman)
 (:objects 
      shaker282 - shaker
      left right - hand
      shot397 shot121 - shot
      ingredient133 ingredient45 - ingredient
      cocktail298 - cocktail
      dispenser149 dispenser4 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker282)
  (ontable shot397)
  (ontable shot121)
  (dispenses dispenser149 ingredient133)
  (dispenses dispenser4 ingredient45)
  (clean shaker282)
  (clean shot397)
  (clean shot121)
  (empty shaker282)
  (empty shot397)
  (empty shot121)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker282 l0)
  (shaker-level shaker282 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail298 ingredient45)
  (cocktail-part2 cocktail298 ingredient133)
)
 (:goal
  (and
      (contains shot397 cocktail298)
)))
