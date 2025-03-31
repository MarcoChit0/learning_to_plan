(define (problem prob)
 (:domain barman)
 (:objects 
      shaker85 - shaker
      left right - hand
      shot53 - shot
      ingredient406 ingredient165 ingredient201 - ingredient
      cocktail1 - cocktail
      dispenser400 dispenser455 dispenser415 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker85)
  (ontable shot53)
  (dispenses dispenser400 ingredient406)
  (dispenses dispenser455 ingredient165)
  (dispenses dispenser415 ingredient201)
  (clean shaker85)
  (clean shot53)
  (empty shaker85)
  (empty shot53)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker85 l0)
  (shaker-level shaker85 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient201)
  (cocktail-part2 cocktail1 ingredient406)
)
 (:goal
  (and
      (contains shot53 cocktail1)
)))
