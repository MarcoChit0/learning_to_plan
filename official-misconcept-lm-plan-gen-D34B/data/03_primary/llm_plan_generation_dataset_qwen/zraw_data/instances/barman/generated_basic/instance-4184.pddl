(define (problem prob)
 (:domain barman)
 (:objects 
      shaker380 - shaker
      left right - hand
      shot290 - shot
      ingredient233 ingredient474 ingredient477 ingredient213 - ingredient
      cocktail23 - cocktail
      dispenser169 dispenser227 dispenser487 dispenser166 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker380)
  (ontable shot290)
  (dispenses dispenser169 ingredient233)
  (dispenses dispenser227 ingredient474)
  (dispenses dispenser487 ingredient477)
  (dispenses dispenser166 ingredient213)
  (clean shaker380)
  (clean shot290)
  (empty shaker380)
  (empty shot290)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker380 l0)
  (shaker-level shaker380 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail23 ingredient213)
  (cocktail-part2 cocktail23 ingredient233)
)
 (:goal
  (and
      (contains shot290 cocktail23)
)))
