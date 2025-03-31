(define (problem prob)
 (:domain barman)
 (:objects 
      shaker103 - shaker
      left right - hand
      shot82 - shot
      ingredient307 ingredient376 - ingredient
      cocktail72 - cocktail
      dispenser17 dispenser295 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker103)
  (ontable shot82)
  (dispenses dispenser17 ingredient307)
  (dispenses dispenser295 ingredient376)
  (clean shaker103)
  (clean shot82)
  (empty shaker103)
  (empty shot82)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker103 l0)
  (shaker-level shaker103 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail72 ingredient376)
  (cocktail-part2 cocktail72 ingredient307)
)
 (:goal
  (and
      (contains shot82 cocktail72)
)))
