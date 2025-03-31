(define (problem prob)
 (:domain barman)
 (:objects 
      shaker402 - shaker
      left right - hand
      shot9 shot294 - shot
      ingredient350 ingredient237 ingredient291 - ingredient
      cocktail186 - cocktail
      dispenser127 dispenser209 dispenser296 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker402)
  (ontable shot9)
  (ontable shot294)
  (dispenses dispenser127 ingredient350)
  (dispenses dispenser209 ingredient237)
  (dispenses dispenser296 ingredient291)
  (clean shaker402)
  (clean shot9)
  (clean shot294)
  (empty shaker402)
  (empty shot9)
  (empty shot294)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker402 l0)
  (shaker-level shaker402 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail186 ingredient350)
  (cocktail-part2 cocktail186 ingredient237)
)
 (:goal
  (and
      (contains shot9 cocktail186)
)))
