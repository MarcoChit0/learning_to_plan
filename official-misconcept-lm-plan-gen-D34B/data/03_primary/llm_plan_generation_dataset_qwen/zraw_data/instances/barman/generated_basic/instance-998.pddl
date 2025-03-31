(define (problem prob)
 (:domain barman)
 (:objects 
      shaker98 - shaker
      left right - hand
      shot415 shot496 - shot
      ingredient312 ingredient63 - ingredient
      cocktail1 - cocktail
      dispenser377 dispenser136 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker98)
  (ontable shot415)
  (ontable shot496)
  (dispenses dispenser377 ingredient312)
  (dispenses dispenser136 ingredient63)
  (clean shaker98)
  (clean shot415)
  (clean shot496)
  (empty shaker98)
  (empty shot415)
  (empty shot496)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker98 l0)
  (shaker-level shaker98 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient312)
  (cocktail-part2 cocktail1 ingredient63)
)
 (:goal
  (and
      (contains shot415 cocktail1)
)))
