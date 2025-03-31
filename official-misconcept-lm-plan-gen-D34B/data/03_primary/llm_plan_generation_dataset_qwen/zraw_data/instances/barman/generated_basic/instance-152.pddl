(define (problem prob)
 (:domain barman)
 (:objects 
      shaker148 - shaker
      left right - hand
      shot90 - shot
      ingredient242 ingredient369 ingredient335 ingredient350 - ingredient
      cocktail1 - cocktail
      dispenser395 dispenser178 dispenser496 dispenser4 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker148)
  (ontable shot90)
  (dispenses dispenser395 ingredient242)
  (dispenses dispenser178 ingredient369)
  (dispenses dispenser496 ingredient335)
  (dispenses dispenser4 ingredient350)
  (clean shaker148)
  (clean shot90)
  (empty shaker148)
  (empty shot90)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker148 l0)
  (shaker-level shaker148 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient242)
  (cocktail-part2 cocktail1 ingredient350)
)
 (:goal
  (and
      (contains shot90 cocktail1)
)))
