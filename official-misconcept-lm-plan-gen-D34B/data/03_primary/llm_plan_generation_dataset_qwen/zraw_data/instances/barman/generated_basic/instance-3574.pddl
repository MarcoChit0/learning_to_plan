(define (problem prob)
 (:domain barman)
 (:objects 
      shaker120 - shaker
      left right - hand
      shot306 shot323 - shot
      ingredient241 ingredient387 ingredient194 ingredient41 - ingredient
      cocktail471 - cocktail
      dispenser410 dispenser192 dispenser113 dispenser115 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker120)
  (ontable shot306)
  (ontable shot323)
  (dispenses dispenser410 ingredient241)
  (dispenses dispenser192 ingredient387)
  (dispenses dispenser113 ingredient194)
  (dispenses dispenser115 ingredient41)
  (clean shaker120)
  (clean shot306)
  (clean shot323)
  (empty shaker120)
  (empty shot306)
  (empty shot323)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker120 l0)
  (shaker-level shaker120 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail471 ingredient241)
  (cocktail-part2 cocktail471 ingredient41)
)
 (:goal
  (and
      (contains shot306 cocktail471)
)))
