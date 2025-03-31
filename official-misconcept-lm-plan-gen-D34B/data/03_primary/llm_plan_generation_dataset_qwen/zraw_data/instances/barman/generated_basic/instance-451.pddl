(define (problem prob)
 (:domain barman)
 (:objects 
      shaker286 - shaker
      left right - hand
      shot27 shot370 - shot
      ingredient382 ingredient257 ingredient323 - ingredient
      cocktail1 - cocktail
      dispenser113 dispenser2 dispenser411 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker286)
  (ontable shot27)
  (ontable shot370)
  (dispenses dispenser113 ingredient382)
  (dispenses dispenser2 ingredient257)
  (dispenses dispenser411 ingredient323)
  (clean shaker286)
  (clean shot27)
  (clean shot370)
  (empty shaker286)
  (empty shot27)
  (empty shot370)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker286 l0)
  (shaker-level shaker286 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient382)
  (cocktail-part2 cocktail1 ingredient323)
)
 (:goal
  (and
      (contains shot27 cocktail1)
)))
