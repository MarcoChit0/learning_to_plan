(define (problem prob)
 (:domain barman)
 (:objects 
      shaker306 - shaker
      left right - hand
      shot366 - shot
      ingredient376 ingredient468 - ingredient
      cocktail164 - cocktail
      dispenser78 dispenser263 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker306)
  (ontable shot366)
  (dispenses dispenser78 ingredient376)
  (dispenses dispenser263 ingredient468)
  (clean shaker306)
  (clean shot366)
  (empty shaker306)
  (empty shot366)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker306 l0)
  (shaker-level shaker306 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail164 ingredient376)
  (cocktail-part2 cocktail164 ingredient468)
)
 (:goal
  (and
      (contains shot366 cocktail164)
)))
