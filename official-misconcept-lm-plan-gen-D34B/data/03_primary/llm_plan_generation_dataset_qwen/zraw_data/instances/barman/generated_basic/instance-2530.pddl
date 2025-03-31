(define (problem prob)
 (:domain barman)
 (:objects 
      shaker303 - shaker
      left right - hand
      shot193 shot61 shot366 - shot
      ingredient414 ingredient134 - ingredient
      cocktail291 - cocktail
      dispenser419 dispenser252 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker303)
  (ontable shot193)
  (ontable shot61)
  (ontable shot366)
  (dispenses dispenser419 ingredient414)
  (dispenses dispenser252 ingredient134)
  (clean shaker303)
  (clean shot193)
  (clean shot61)
  (clean shot366)
  (empty shaker303)
  (empty shot193)
  (empty shot61)
  (empty shot366)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker303 l0)
  (shaker-level shaker303 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail291 ingredient134)
  (cocktail-part2 cocktail291 ingredient414)
)
 (:goal
  (and
      (contains shot193 cocktail291)
      (contains shot61 cocktail291)
)))
