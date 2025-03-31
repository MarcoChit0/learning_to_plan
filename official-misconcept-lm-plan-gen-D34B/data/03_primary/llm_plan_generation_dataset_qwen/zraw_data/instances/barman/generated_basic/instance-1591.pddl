(define (problem prob)
 (:domain barman)
 (:objects 
      shaker0 - shaker
      left right - hand
      shot6 shot33 - shot
      ingredient103 ingredient65 - ingredient
      cocktail88 - cocktail
      dispenser248 dispenser59 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker0)
  (ontable shot6)
  (ontable shot33)
  (dispenses dispenser248 ingredient103)
  (dispenses dispenser59 ingredient65)
  (clean shaker0)
  (clean shot6)
  (clean shot33)
  (empty shaker0)
  (empty shot6)
  (empty shot33)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker0 l0)
  (shaker-level shaker0 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail88 ingredient103)
  (cocktail-part2 cocktail88 ingredient65)
)
 (:goal
  (and
      (contains shot6 cocktail88)
)))
