(define (problem prob)
 (:domain barman)
 (:objects 
      shaker230 - shaker
      left right - hand
      shot369 - shot
      ingredient326 ingredient468 - ingredient
      cocktail1 - cocktail
      dispenser19 dispenser465 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker230)
  (ontable shot369)
  (dispenses dispenser19 ingredient326)
  (dispenses dispenser465 ingredient468)
  (clean shaker230)
  (clean shot369)
  (empty shaker230)
  (empty shot369)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker230 l0)
  (shaker-level shaker230 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient468)
  (cocktail-part2 cocktail1 ingredient326)
)
 (:goal
  (and
      (contains shot369 cocktail1)
)))
