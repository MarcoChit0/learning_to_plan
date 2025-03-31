(define (problem prob)
 (:domain barman)
 (:objects 
      shaker94 - shaker
      left right - hand
      shot344 shot302 shot381 - shot
      ingredient198 ingredient233 - ingredient
      cocktail94 - cocktail
      dispenser486 dispenser133 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker94)
  (ontable shot344)
  (ontable shot302)
  (ontable shot381)
  (dispenses dispenser486 ingredient198)
  (dispenses dispenser133 ingredient233)
  (clean shaker94)
  (clean shot344)
  (clean shot302)
  (clean shot381)
  (empty shaker94)
  (empty shot344)
  (empty shot302)
  (empty shot381)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker94 l0)
  (shaker-level shaker94 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail94 ingredient198)
  (cocktail-part2 cocktail94 ingredient233)
)
 (:goal
  (and
      (contains shot344 cocktail94)
      (contains shot302 ingredient198)
)))
