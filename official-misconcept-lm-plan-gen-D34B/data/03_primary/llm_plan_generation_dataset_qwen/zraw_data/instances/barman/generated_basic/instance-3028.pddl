(define (problem prob)
 (:domain barman)
 (:objects 
      shaker483 - shaker
      left right - hand
      shot59 - shot
      ingredient431 ingredient335 ingredient463 - ingredient
      cocktail190 - cocktail
      dispenser415 dispenser457 dispenser445 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker483)
  (ontable shot59)
  (dispenses dispenser415 ingredient431)
  (dispenses dispenser457 ingredient335)
  (dispenses dispenser445 ingredient463)
  (clean shaker483)
  (clean shot59)
  (empty shaker483)
  (empty shot59)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker483 l0)
  (shaker-level shaker483 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail190 ingredient431)
  (cocktail-part2 cocktail190 ingredient335)
)
 (:goal
  (and
      (contains shot59 cocktail190)
)))
