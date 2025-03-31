(define (problem prob)
 (:domain barman)
 (:objects 
      shaker78 - shaker
      left right - hand
      shot428 shot486 - shot
      ingredient62 ingredient226 ingredient446 ingredient60 - ingredient
      cocktail184 - cocktail
      dispenser87 dispenser276 dispenser133 dispenser162 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker78)
  (ontable shot428)
  (ontable shot486)
  (dispenses dispenser87 ingredient62)
  (dispenses dispenser276 ingredient226)
  (dispenses dispenser133 ingredient446)
  (dispenses dispenser162 ingredient60)
  (clean shaker78)
  (clean shot428)
  (clean shot486)
  (empty shaker78)
  (empty shot428)
  (empty shot486)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker78 l0)
  (shaker-level shaker78 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail184 ingredient226)
  (cocktail-part2 cocktail184 ingredient60)
)
 (:goal
  (and
      (contains shot428 cocktail184)
)))
