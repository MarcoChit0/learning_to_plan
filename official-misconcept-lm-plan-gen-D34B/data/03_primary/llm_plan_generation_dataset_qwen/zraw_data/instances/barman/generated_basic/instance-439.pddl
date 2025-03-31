(define (problem prob)
 (:domain barman)
 (:objects 
      shaker450 - shaker
      left right - hand
      shot340 shot386 - shot
      ingredient247 ingredient136 ingredient41 - ingredient
      cocktail1 - cocktail
      dispenser362 dispenser338 dispenser123 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker450)
  (ontable shot340)
  (ontable shot386)
  (dispenses dispenser362 ingredient247)
  (dispenses dispenser338 ingredient136)
  (dispenses dispenser123 ingredient41)
  (clean shaker450)
  (clean shot340)
  (clean shot386)
  (empty shaker450)
  (empty shot340)
  (empty shot386)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker450 l0)
  (shaker-level shaker450 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient41)
  (cocktail-part2 cocktail1 ingredient247)
)
 (:goal
  (and
      (contains shot340 cocktail1)
)))
