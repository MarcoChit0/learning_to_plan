(define (problem prob)
 (:domain barman)
 (:objects 
      shaker426 - shaker
      left right - hand
      shot371 - shot
      ingredient60 ingredient138 - ingredient
      cocktail331 - cocktail
      dispenser175 dispenser98 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker426)
  (ontable shot371)
  (dispenses dispenser175 ingredient60)
  (dispenses dispenser98 ingredient138)
  (clean shaker426)
  (clean shot371)
  (empty shaker426)
  (empty shot371)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker426 l0)
  (shaker-level shaker426 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail331 ingredient138)
  (cocktail-part2 cocktail331 ingredient60)
)
 (:goal
  (and
      (contains shot371 cocktail331)
)))
