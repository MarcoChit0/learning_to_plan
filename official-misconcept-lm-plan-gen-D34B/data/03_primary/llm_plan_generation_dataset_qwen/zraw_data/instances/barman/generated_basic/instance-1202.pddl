(define (problem prob)
 (:domain barman)
 (:objects 
      shaker47 - shaker
      left right - hand
      shot480 shot117 - shot
      ingredient406 ingredient5 - ingredient
      cocktail343 - cocktail
      dispenser384 dispenser274 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker47)
  (ontable shot480)
  (ontable shot117)
  (dispenses dispenser384 ingredient406)
  (dispenses dispenser274 ingredient5)
  (clean shaker47)
  (clean shot480)
  (clean shot117)
  (empty shaker47)
  (empty shot480)
  (empty shot117)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker47 l0)
  (shaker-level shaker47 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail343 ingredient5)
  (cocktail-part2 cocktail343 ingredient406)
)
 (:goal
  (and
      (contains shot480 cocktail343)
)))
