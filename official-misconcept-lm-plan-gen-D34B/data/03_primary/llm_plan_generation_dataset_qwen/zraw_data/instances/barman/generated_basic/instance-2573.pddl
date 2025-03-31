(define (problem prob)
 (:domain barman)
 (:objects 
      shaker91 - shaker
      left right - hand
      shot290 - shot
      ingredient204 ingredient331 - ingredient
      cocktail120 - cocktail
      dispenser458 dispenser389 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker91)
  (ontable shot290)
  (dispenses dispenser458 ingredient204)
  (dispenses dispenser389 ingredient331)
  (clean shaker91)
  (clean shot290)
  (empty shaker91)
  (empty shot290)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker91 l0)
  (shaker-level shaker91 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail120 ingredient204)
  (cocktail-part2 cocktail120 ingredient331)
)
 (:goal
  (and
      (contains shot290 cocktail120)
)))
