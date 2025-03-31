(define (problem prob)
 (:domain barman)
 (:objects 
      shaker276 - shaker
      left right - hand
      shot191 - shot
      ingredient497 ingredient442 ingredient123 - ingredient
      cocktail243 - cocktail
      dispenser146 dispenser468 dispenser49 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker276)
  (ontable shot191)
  (dispenses dispenser146 ingredient497)
  (dispenses dispenser468 ingredient442)
  (dispenses dispenser49 ingredient123)
  (clean shaker276)
  (clean shot191)
  (empty shaker276)
  (empty shot191)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker276 l0)
  (shaker-level shaker276 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail243 ingredient497)
  (cocktail-part2 cocktail243 ingredient123)
)
 (:goal
  (and
      (contains shot191 cocktail243)
)))
