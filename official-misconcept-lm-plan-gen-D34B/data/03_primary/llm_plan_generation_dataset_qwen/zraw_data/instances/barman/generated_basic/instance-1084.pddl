(define (problem prob)
 (:domain barman)
 (:objects 
      shaker306 - shaker
      left right - hand
      shot320 - shot
      ingredient296 ingredient357 ingredient8 - ingredient
      cocktail1 - cocktail
      dispenser424 dispenser449 dispenser211 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker306)
  (ontable shot320)
  (dispenses dispenser424 ingredient296)
  (dispenses dispenser449 ingredient357)
  (dispenses dispenser211 ingredient8)
  (clean shaker306)
  (clean shot320)
  (empty shaker306)
  (empty shot320)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker306 l0)
  (shaker-level shaker306 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient357)
  (cocktail-part2 cocktail1 ingredient296)
)
 (:goal
  (and
      (contains shot320 cocktail1)
)))
