(define (problem prob)
 (:domain barman)
 (:objects 
      shaker197 - shaker
      left right - hand
      shot171 - shot
      ingredient4 ingredient231 - ingredient
      cocktail1 - cocktail
      dispenser256 dispenser342 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker197)
  (ontable shot171)
  (dispenses dispenser256 ingredient4)
  (dispenses dispenser342 ingredient231)
  (clean shaker197)
  (clean shot171)
  (empty shaker197)
  (empty shot171)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker197 l0)
  (shaker-level shaker197 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient231)
  (cocktail-part2 cocktail1 ingredient4)
)
 (:goal
  (and
      (contains shot171 cocktail1)
)))
