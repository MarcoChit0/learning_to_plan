(define (problem prob)
 (:domain barman)
 (:objects 
      shaker159 - shaker
      left right - hand
      shot7 - shot
      ingredient172 ingredient115 ingredient128 ingredient54 - ingredient
      cocktail23 - cocktail
      dispenser453 dispenser329 dispenser279 dispenser145 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker159)
  (ontable shot7)
  (dispenses dispenser453 ingredient172)
  (dispenses dispenser329 ingredient115)
  (dispenses dispenser279 ingredient128)
  (dispenses dispenser145 ingredient54)
  (clean shaker159)
  (clean shot7)
  (empty shaker159)
  (empty shot7)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker159 l0)
  (shaker-level shaker159 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail23 ingredient54)
  (cocktail-part2 cocktail23 ingredient128)
)
 (:goal
  (and
      (contains shot7 cocktail23)
)))
