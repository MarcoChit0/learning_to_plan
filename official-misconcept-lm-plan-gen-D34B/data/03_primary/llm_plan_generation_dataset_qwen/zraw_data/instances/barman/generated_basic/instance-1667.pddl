(define (problem prob)
 (:domain barman)
 (:objects 
      shaker50 - shaker
      left right - hand
      shot426 shot339 - shot
      ingredient6 ingredient134 ingredient386 ingredient311 - ingredient
      cocktail49 - cocktail
      dispenser350 dispenser355 dispenser363 dispenser425 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker50)
  (ontable shot426)
  (ontable shot339)
  (dispenses dispenser350 ingredient6)
  (dispenses dispenser355 ingredient134)
  (dispenses dispenser363 ingredient386)
  (dispenses dispenser425 ingredient311)
  (clean shaker50)
  (clean shot426)
  (clean shot339)
  (empty shaker50)
  (empty shot426)
  (empty shot339)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker50 l0)
  (shaker-level shaker50 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail49 ingredient311)
  (cocktail-part2 cocktail49 ingredient386)
)
 (:goal
  (and
      (contains shot426 cocktail49)
)))
