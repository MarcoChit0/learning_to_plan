(define (problem prob)
 (:domain barman)
 (:objects 
      shaker481 - shaker
      left right - hand
      shot463 shot396 - shot
      ingredient184 ingredient220 ingredient427 ingredient354 - ingredient
      cocktail413 - cocktail
      dispenser280 dispenser480 dispenser160 dispenser340 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker481)
  (ontable shot463)
  (ontable shot396)
  (dispenses dispenser280 ingredient184)
  (dispenses dispenser480 ingredient220)
  (dispenses dispenser160 ingredient427)
  (dispenses dispenser340 ingredient354)
  (clean shaker481)
  (clean shot463)
  (clean shot396)
  (empty shaker481)
  (empty shot463)
  (empty shot396)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker481 l0)
  (shaker-level shaker481 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail413 ingredient354)
  (cocktail-part2 cocktail413 ingredient427)
)
 (:goal
  (and
      (contains shot463 cocktail413)
)))
