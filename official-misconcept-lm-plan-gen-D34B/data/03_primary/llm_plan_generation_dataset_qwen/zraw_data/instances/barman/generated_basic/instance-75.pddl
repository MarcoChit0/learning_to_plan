(define (problem prob)
 (:domain barman)
 (:objects 
      shaker406 - shaker
      left right - hand
      shot6 shot216 - shot
      ingredient390 ingredient142 ingredient459 - ingredient
      cocktail1 - cocktail
      dispenser173 dispenser405 dispenser490 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker406)
  (ontable shot6)
  (ontable shot216)
  (dispenses dispenser173 ingredient390)
  (dispenses dispenser405 ingredient142)
  (dispenses dispenser490 ingredient459)
  (clean shaker406)
  (clean shot6)
  (clean shot216)
  (empty shaker406)
  (empty shot6)
  (empty shot216)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker406 l0)
  (shaker-level shaker406 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient142)
  (cocktail-part2 cocktail1 ingredient390)
)
 (:goal
  (and
      (contains shot6 cocktail1)
)))
