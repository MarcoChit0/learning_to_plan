(define (problem prob)
 (:domain barman)
 (:objects 
      shaker311 - shaker
      left right - hand
      shot388 - shot
      ingredient200 ingredient484 ingredient16 - ingredient
      cocktail1 - cocktail
      dispenser496 dispenser57 dispenser3 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker311)
  (ontable shot388)
  (dispenses dispenser496 ingredient200)
  (dispenses dispenser57 ingredient484)
  (dispenses dispenser3 ingredient16)
  (clean shaker311)
  (clean shot388)
  (empty shaker311)
  (empty shot388)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker311 l0)
  (shaker-level shaker311 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient484)
  (cocktail-part2 cocktail1 ingredient16)
)
 (:goal
  (and
      (contains shot388 cocktail1)
)))
