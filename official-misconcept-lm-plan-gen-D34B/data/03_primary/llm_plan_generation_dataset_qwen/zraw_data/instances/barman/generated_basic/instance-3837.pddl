(define (problem prob)
 (:domain barman)
 (:objects 
      shaker62 - shaker
      left right - hand
      shot268 shot425 shot59 - shot
      ingredient490 ingredient221 - ingredient
      cocktail402 - cocktail
      dispenser390 dispenser442 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker62)
  (ontable shot268)
  (ontable shot425)
  (ontable shot59)
  (dispenses dispenser390 ingredient490)
  (dispenses dispenser442 ingredient221)
  (clean shaker62)
  (clean shot268)
  (clean shot425)
  (clean shot59)
  (empty shaker62)
  (empty shot268)
  (empty shot425)
  (empty shot59)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker62 l0)
  (shaker-level shaker62 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail402 ingredient221)
  (cocktail-part2 cocktail402 ingredient490)
)
 (:goal
  (and
      (contains shot268 cocktail402)
      (contains shot425 cocktail402)
)))
