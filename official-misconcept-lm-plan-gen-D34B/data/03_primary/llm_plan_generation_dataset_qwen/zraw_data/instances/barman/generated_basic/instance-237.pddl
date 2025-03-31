(define (problem prob)
 (:domain barman)
 (:objects 
      shaker448 - shaker
      left right - hand
      shot349 - shot
      ingredient352 ingredient209 - ingredient
      cocktail1 - cocktail
      dispenser107 dispenser442 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker448)
  (ontable shot349)
  (dispenses dispenser107 ingredient352)
  (dispenses dispenser442 ingredient209)
  (clean shaker448)
  (clean shot349)
  (empty shaker448)
  (empty shot349)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker448 l0)
  (shaker-level shaker448 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient352)
  (cocktail-part2 cocktail1 ingredient209)
)
 (:goal
  (and
      (contains shot349 cocktail1)
)))
