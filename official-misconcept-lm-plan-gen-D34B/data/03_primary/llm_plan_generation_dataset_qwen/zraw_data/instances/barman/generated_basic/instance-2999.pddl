(define (problem prob)
 (:domain barman)
 (:objects 
      shaker259 - shaker
      left right - hand
      shot259 shot218 - shot
      ingredient276 ingredient288 ingredient426 - ingredient
      cocktail479 - cocktail
      dispenser286 dispenser340 dispenser77 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker259)
  (ontable shot259)
  (ontable shot218)
  (dispenses dispenser286 ingredient276)
  (dispenses dispenser340 ingredient288)
  (dispenses dispenser77 ingredient426)
  (clean shaker259)
  (clean shot259)
  (clean shot218)
  (empty shaker259)
  (empty shot259)
  (empty shot218)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker259 l0)
  (shaker-level shaker259 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail479 ingredient288)
  (cocktail-part2 cocktail479 ingredient426)
)
 (:goal
  (and
      (contains shot259 cocktail479)
)))
