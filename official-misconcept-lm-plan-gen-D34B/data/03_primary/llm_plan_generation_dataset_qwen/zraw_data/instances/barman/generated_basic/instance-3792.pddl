(define (problem prob)
 (:domain barman)
 (:objects 
      shaker440 - shaker
      left right - hand
      shot234 - shot
      ingredient194 ingredient343 ingredient463 ingredient171 - ingredient
      cocktail480 - cocktail
      dispenser196 dispenser232 dispenser225 dispenser437 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker440)
  (ontable shot234)
  (dispenses dispenser196 ingredient194)
  (dispenses dispenser232 ingredient343)
  (dispenses dispenser225 ingredient463)
  (dispenses dispenser437 ingredient171)
  (clean shaker440)
  (clean shot234)
  (empty shaker440)
  (empty shot234)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker440 l0)
  (shaker-level shaker440 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail480 ingredient194)
  (cocktail-part2 cocktail480 ingredient171)
)
 (:goal
  (and
      (contains shot234 cocktail480)
)))
