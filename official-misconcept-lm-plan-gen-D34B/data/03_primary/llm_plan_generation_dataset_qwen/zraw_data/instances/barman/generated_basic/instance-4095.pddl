(define (problem prob)
 (:domain barman)
 (:objects 
      shaker171 - shaker
      left right - hand
      shot271 shot237 - shot
      ingredient218 ingredient402 - ingredient
      cocktail155 - cocktail
      dispenser74 dispenser101 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker171)
  (ontable shot271)
  (ontable shot237)
  (dispenses dispenser74 ingredient218)
  (dispenses dispenser101 ingredient402)
  (clean shaker171)
  (clean shot271)
  (clean shot237)
  (empty shaker171)
  (empty shot271)
  (empty shot237)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker171 l0)
  (shaker-level shaker171 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail155 ingredient218)
  (cocktail-part2 cocktail155 ingredient402)
)
 (:goal
  (and
      (contains shot271 cocktail155)
)))
