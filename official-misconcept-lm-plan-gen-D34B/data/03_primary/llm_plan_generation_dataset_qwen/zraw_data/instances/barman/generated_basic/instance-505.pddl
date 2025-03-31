(define (problem prob)
 (:domain barman)
 (:objects 
      shaker167 - shaker
      left right - hand
      shot30 shot248 - shot
      ingredient61 ingredient489 ingredient293 ingredient463 - ingredient
      cocktail1 - cocktail
      dispenser68 dispenser252 dispenser201 dispenser15 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker167)
  (ontable shot30)
  (ontable shot248)
  (dispenses dispenser68 ingredient61)
  (dispenses dispenser252 ingredient489)
  (dispenses dispenser201 ingredient293)
  (dispenses dispenser15 ingredient463)
  (clean shaker167)
  (clean shot30)
  (clean shot248)
  (empty shaker167)
  (empty shot30)
  (empty shot248)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker167 l0)
  (shaker-level shaker167 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient463)
  (cocktail-part2 cocktail1 ingredient61)
)
 (:goal
  (and
      (contains shot30 cocktail1)
)))
