(define (problem prob)
 (:domain barman)
 (:objects 
      shaker7 - shaker
      left right - hand
      shot90 shot250 - shot
      ingredient205 ingredient64 ingredient160 - ingredient
      cocktail1 - cocktail
      dispenser110 dispenser77 dispenser475 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker7)
  (ontable shot90)
  (ontable shot250)
  (dispenses dispenser110 ingredient205)
  (dispenses dispenser77 ingredient64)
  (dispenses dispenser475 ingredient160)
  (clean shaker7)
  (clean shot90)
  (clean shot250)
  (empty shaker7)
  (empty shot90)
  (empty shot250)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker7 l0)
  (shaker-level shaker7 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient64)
  (cocktail-part2 cocktail1 ingredient205)
)
 (:goal
  (and
      (contains shot90 cocktail1)
)))
