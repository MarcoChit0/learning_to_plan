(define (problem prob)
 (:domain barman)
 (:objects 
      shaker382 - shaker
      left right - hand
      shot290 - shot
      ingredient469 ingredient238 - ingredient
      cocktail146 - cocktail
      dispenser227 dispenser312 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker382)
  (ontable shot290)
  (dispenses dispenser227 ingredient469)
  (dispenses dispenser312 ingredient238)
  (clean shaker382)
  (clean shot290)
  (empty shaker382)
  (empty shot290)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker382 l0)
  (shaker-level shaker382 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail146 ingredient238)
  (cocktail-part2 cocktail146 ingredient469)
)
 (:goal
  (and
      (contains shot290 cocktail146)
)))
