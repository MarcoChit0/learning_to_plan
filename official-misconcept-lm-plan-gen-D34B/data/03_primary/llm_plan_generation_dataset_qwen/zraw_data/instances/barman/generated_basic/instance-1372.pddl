(define (problem prob)
 (:domain barman)
 (:objects 
      shaker228 - shaker
      left right - hand
      shot262 - shot
      ingredient258 ingredient180 ingredient224 - ingredient
      cocktail55 - cocktail
      dispenser196 dispenser448 dispenser330 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker228)
  (ontable shot262)
  (dispenses dispenser196 ingredient258)
  (dispenses dispenser448 ingredient180)
  (dispenses dispenser330 ingredient224)
  (clean shaker228)
  (clean shot262)
  (empty shaker228)
  (empty shot262)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker228 l0)
  (shaker-level shaker228 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail55 ingredient180)
  (cocktail-part2 cocktail55 ingredient258)
)
 (:goal
  (and
      (contains shot262 cocktail55)
)))
