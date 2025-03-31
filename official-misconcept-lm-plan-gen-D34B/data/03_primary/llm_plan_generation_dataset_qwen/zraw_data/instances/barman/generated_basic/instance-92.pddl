(define (problem prob)
 (:domain barman)
 (:objects 
      shaker198 - shaker
      left right - hand
      shot117 - shot
      ingredient120 ingredient429 ingredient269 ingredient463 - ingredient
      cocktail1 - cocktail
      dispenser390 dispenser108 dispenser495 dispenser300 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker198)
  (ontable shot117)
  (dispenses dispenser390 ingredient120)
  (dispenses dispenser108 ingredient429)
  (dispenses dispenser495 ingredient269)
  (dispenses dispenser300 ingredient463)
  (clean shaker198)
  (clean shot117)
  (empty shaker198)
  (empty shot117)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker198 l0)
  (shaker-level shaker198 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient463)
  (cocktail-part2 cocktail1 ingredient120)
)
 (:goal
  (and
      (contains shot117 cocktail1)
)))
