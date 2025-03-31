(define (problem prob)
 (:domain barman)
 (:objects 
      shaker452 - shaker
      left right - hand
      shot190 shot48 - shot
      ingredient205 ingredient339 ingredient330 ingredient123 - ingredient
      cocktail1 - cocktail
      dispenser486 dispenser475 dispenser188 dispenser213 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker452)
  (ontable shot190)
  (ontable shot48)
  (dispenses dispenser486 ingredient205)
  (dispenses dispenser475 ingredient339)
  (dispenses dispenser188 ingredient330)
  (dispenses dispenser213 ingredient123)
  (clean shaker452)
  (clean shot190)
  (clean shot48)
  (empty shaker452)
  (empty shot190)
  (empty shot48)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker452 l0)
  (shaker-level shaker452 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient123)
  (cocktail-part2 cocktail1 ingredient339)
)
 (:goal
  (and
      (contains shot190 cocktail1)
)))
