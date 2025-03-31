(define (problem prob)
 (:domain barman)
 (:objects 
      shaker17 - shaker
      left right - hand
      shot222 shot227 - shot
      ingredient428 ingredient79 - ingredient
      cocktail35 - cocktail
      dispenser396 dispenser106 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker17)
  (ontable shot222)
  (ontable shot227)
  (dispenses dispenser396 ingredient428)
  (dispenses dispenser106 ingredient79)
  (clean shaker17)
  (clean shot222)
  (clean shot227)
  (empty shaker17)
  (empty shot222)
  (empty shot227)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker17 l0)
  (shaker-level shaker17 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail35 ingredient428)
  (cocktail-part2 cocktail35 ingredient79)
)
 (:goal
  (and
      (contains shot222 cocktail35)
)))
