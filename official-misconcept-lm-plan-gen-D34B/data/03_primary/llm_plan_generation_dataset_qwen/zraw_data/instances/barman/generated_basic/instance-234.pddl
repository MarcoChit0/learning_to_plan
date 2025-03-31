(define (problem prob)
 (:domain barman)
 (:objects 
      shaker84 - shaker
      left right - hand
      shot134 - shot
      ingredient261 ingredient107 - ingredient
      cocktail1 - cocktail
      dispenser377 dispenser305 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker84)
  (ontable shot134)
  (dispenses dispenser377 ingredient261)
  (dispenses dispenser305 ingredient107)
  (clean shaker84)
  (clean shot134)
  (empty shaker84)
  (empty shot134)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker84 l0)
  (shaker-level shaker84 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient261)
  (cocktail-part2 cocktail1 ingredient107)
)
 (:goal
  (and
      (contains shot134 cocktail1)
)))
