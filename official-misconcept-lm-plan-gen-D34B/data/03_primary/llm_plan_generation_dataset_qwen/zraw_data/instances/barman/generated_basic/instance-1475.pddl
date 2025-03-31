(define (problem prob)
 (:domain barman)
 (:objects 
      shaker486 - shaker
      left right - hand
      shot218 shot21 - shot
      ingredient306 ingredient116 ingredient185 ingredient196 - ingredient
      cocktail431 - cocktail
      dispenser257 dispenser25 dispenser75 dispenser74 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker486)
  (ontable shot218)
  (ontable shot21)
  (dispenses dispenser257 ingredient306)
  (dispenses dispenser25 ingredient116)
  (dispenses dispenser75 ingredient185)
  (dispenses dispenser74 ingredient196)
  (clean shaker486)
  (clean shot218)
  (clean shot21)
  (empty shaker486)
  (empty shot218)
  (empty shot21)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker486 l0)
  (shaker-level shaker486 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail431 ingredient196)
  (cocktail-part2 cocktail431 ingredient116)
)
 (:goal
  (and
      (contains shot218 cocktail431)
)))
