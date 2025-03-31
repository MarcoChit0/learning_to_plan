(define (problem prob)
 (:domain barman)
 (:objects 
      shaker99 - shaker
      left right - hand
      shot29 shot217 - shot
      ingredient204 ingredient342 ingredient421 - ingredient
      cocktail1 - cocktail
      dispenser289 dispenser359 dispenser202 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker99)
  (ontable shot29)
  (ontable shot217)
  (dispenses dispenser289 ingredient204)
  (dispenses dispenser359 ingredient342)
  (dispenses dispenser202 ingredient421)
  (clean shaker99)
  (clean shot29)
  (clean shot217)
  (empty shaker99)
  (empty shot29)
  (empty shot217)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker99 l0)
  (shaker-level shaker99 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient204)
  (cocktail-part2 cocktail1 ingredient421)
)
 (:goal
  (and
      (contains shot29 cocktail1)
)))
