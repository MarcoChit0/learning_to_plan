(define (problem prob)
 (:domain barman)
 (:objects 
      shaker371 - shaker
      left right - hand
      shot222 shot254 - shot
      ingredient244 ingredient455 ingredient281 - ingredient
      cocktail24 - cocktail
      dispenser279 dispenser374 dispenser293 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker371)
  (ontable shot222)
  (ontable shot254)
  (dispenses dispenser279 ingredient244)
  (dispenses dispenser374 ingredient455)
  (dispenses dispenser293 ingredient281)
  (clean shaker371)
  (clean shot222)
  (clean shot254)
  (empty shaker371)
  (empty shot222)
  (empty shot254)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker371 l0)
  (shaker-level shaker371 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail24 ingredient244)
  (cocktail-part2 cocktail24 ingredient455)
)
 (:goal
  (and
      (contains shot222 cocktail24)
)))
