(define (problem prob)
 (:domain barman)
 (:objects 
      shaker276 - shaker
      left right - hand
      shot68 shot319 - shot
      ingredient160 ingredient337 - ingredient
      cocktail484 - cocktail
      dispenser271 dispenser139 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker276)
  (ontable shot68)
  (ontable shot319)
  (dispenses dispenser271 ingredient160)
  (dispenses dispenser139 ingredient337)
  (clean shaker276)
  (clean shot68)
  (clean shot319)
  (empty shaker276)
  (empty shot68)
  (empty shot319)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker276 l0)
  (shaker-level shaker276 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail484 ingredient337)
  (cocktail-part2 cocktail484 ingredient160)
)
 (:goal
  (and
      (contains shot68 cocktail484)
)))
