(define (problem prob)
 (:domain barman)
 (:objects 
      shaker300 - shaker
      left right - hand
      shot297 shot313 - shot
      ingredient348 ingredient25 - ingredient
      cocktail1 - cocktail
      dispenser390 dispenser348 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker300)
  (ontable shot297)
  (ontable shot313)
  (dispenses dispenser390 ingredient348)
  (dispenses dispenser348 ingredient25)
  (clean shaker300)
  (clean shot297)
  (clean shot313)
  (empty shaker300)
  (empty shot297)
  (empty shot313)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker300 l0)
  (shaker-level shaker300 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient25)
  (cocktail-part2 cocktail1 ingredient348)
)
 (:goal
  (and
      (contains shot297 cocktail1)
)))
