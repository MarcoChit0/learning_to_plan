(define (problem prob)
 (:domain barman)
 (:objects 
      shaker17 - shaker
      left right - hand
      shot297 shot125 - shot
      ingredient293 ingredient6 ingredient218 - ingredient
      cocktail1 - cocktail
      dispenser425 dispenser378 dispenser74 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker17)
  (ontable shot297)
  (ontable shot125)
  (dispenses dispenser425 ingredient293)
  (dispenses dispenser378 ingredient6)
  (dispenses dispenser74 ingredient218)
  (clean shaker17)
  (clean shot297)
  (clean shot125)
  (empty shaker17)
  (empty shot297)
  (empty shot125)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker17 l0)
  (shaker-level shaker17 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient218)
  (cocktail-part2 cocktail1 ingredient293)
)
 (:goal
  (and
      (contains shot297 cocktail1)
)))
