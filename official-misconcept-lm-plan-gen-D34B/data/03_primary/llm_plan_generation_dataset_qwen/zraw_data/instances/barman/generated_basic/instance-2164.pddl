(define (problem prob)
 (:domain barman)
 (:objects 
      shaker78 - shaker
      left right - hand
      shot225 shot71 - shot
      ingredient405 ingredient231 - ingredient
      cocktail424 - cocktail
      dispenser433 dispenser165 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker78)
  (ontable shot225)
  (ontable shot71)
  (dispenses dispenser433 ingredient405)
  (dispenses dispenser165 ingredient231)
  (clean shaker78)
  (clean shot225)
  (clean shot71)
  (empty shaker78)
  (empty shot225)
  (empty shot71)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker78 l0)
  (shaker-level shaker78 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail424 ingredient405)
  (cocktail-part2 cocktail424 ingredient231)
)
 (:goal
  (and
      (contains shot225 cocktail424)
)))
