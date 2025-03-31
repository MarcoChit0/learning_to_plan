(define (problem prob)
 (:domain barman)
 (:objects 
      shaker162 - shaker
      left right - hand
      shot227 - shot
      ingredient94 ingredient238 ingredient323 - ingredient
      cocktail1 - cocktail
      dispenser458 dispenser465 dispenser261 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker162)
  (ontable shot227)
  (dispenses dispenser458 ingredient94)
  (dispenses dispenser465 ingredient238)
  (dispenses dispenser261 ingredient323)
  (clean shaker162)
  (clean shot227)
  (empty shaker162)
  (empty shot227)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker162 l0)
  (shaker-level shaker162 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient323)
  (cocktail-part2 cocktail1 ingredient94)
)
 (:goal
  (and
      (contains shot227 cocktail1)
)))
