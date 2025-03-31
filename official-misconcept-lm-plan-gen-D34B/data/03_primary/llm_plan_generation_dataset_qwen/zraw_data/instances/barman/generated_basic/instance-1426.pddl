(define (problem prob)
 (:domain barman)
 (:objects 
      shaker485 - shaker
      left right - hand
      shot95 shot91 - shot
      ingredient79 ingredient142 - ingredient
      cocktail96 - cocktail
      dispenser275 dispenser344 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker485)
  (ontable shot95)
  (ontable shot91)
  (dispenses dispenser275 ingredient79)
  (dispenses dispenser344 ingredient142)
  (clean shaker485)
  (clean shot95)
  (clean shot91)
  (empty shaker485)
  (empty shot95)
  (empty shot91)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker485 l0)
  (shaker-level shaker485 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail96 ingredient79)
  (cocktail-part2 cocktail96 ingredient142)
)
 (:goal
  (and
      (contains shot95 cocktail96)
)))
