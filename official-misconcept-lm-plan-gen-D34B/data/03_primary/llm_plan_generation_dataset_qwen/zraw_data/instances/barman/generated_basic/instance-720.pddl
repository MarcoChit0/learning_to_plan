(define (problem prob)
 (:domain barman)
 (:objects 
      shaker164 - shaker
      left right - hand
      shot271 - shot
      ingredient89 ingredient325 ingredient471 - ingredient
      cocktail1 - cocktail
      dispenser336 dispenser338 dispenser267 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker164)
  (ontable shot271)
  (dispenses dispenser336 ingredient89)
  (dispenses dispenser338 ingredient325)
  (dispenses dispenser267 ingredient471)
  (clean shaker164)
  (clean shot271)
  (empty shaker164)
  (empty shot271)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker164 l0)
  (shaker-level shaker164 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient471)
  (cocktail-part2 cocktail1 ingredient89)
)
 (:goal
  (and
      (contains shot271 cocktail1)
)))
