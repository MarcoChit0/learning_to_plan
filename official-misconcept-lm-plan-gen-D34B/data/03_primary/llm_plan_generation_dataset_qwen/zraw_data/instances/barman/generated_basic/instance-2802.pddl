(define (problem prob)
 (:domain barman)
 (:objects 
      shaker442 - shaker
      left right - hand
      shot250 shot96 - shot
      ingredient72 ingredient110 - ingredient
      cocktail434 - cocktail
      dispenser455 dispenser463 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker442)
  (ontable shot250)
  (ontable shot96)
  (dispenses dispenser455 ingredient72)
  (dispenses dispenser463 ingredient110)
  (clean shaker442)
  (clean shot250)
  (clean shot96)
  (empty shaker442)
  (empty shot250)
  (empty shot96)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker442 l0)
  (shaker-level shaker442 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail434 ingredient72)
  (cocktail-part2 cocktail434 ingredient110)
)
 (:goal
  (and
      (contains shot250 cocktail434)
)))
