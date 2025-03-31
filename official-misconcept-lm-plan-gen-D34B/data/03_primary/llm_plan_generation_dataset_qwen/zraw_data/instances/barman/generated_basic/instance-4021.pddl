(define (problem prob)
 (:domain barman)
 (:objects 
      shaker405 - shaker
      left right - hand
      shot468 shot106 - shot
      ingredient241 ingredient40 ingredient43 - ingredient
      cocktail380 - cocktail
      dispenser223 dispenser482 dispenser130 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker405)
  (ontable shot468)
  (ontable shot106)
  (dispenses dispenser223 ingredient241)
  (dispenses dispenser482 ingredient40)
  (dispenses dispenser130 ingredient43)
  (clean shaker405)
  (clean shot468)
  (clean shot106)
  (empty shaker405)
  (empty shot468)
  (empty shot106)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker405 l0)
  (shaker-level shaker405 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail380 ingredient241)
  (cocktail-part2 cocktail380 ingredient40)
)
 (:goal
  (and
      (contains shot468 cocktail380)
)))
