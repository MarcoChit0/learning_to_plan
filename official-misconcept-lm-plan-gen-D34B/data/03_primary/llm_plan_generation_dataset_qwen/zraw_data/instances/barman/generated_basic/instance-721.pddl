(define (problem prob)
 (:domain barman)
 (:objects 
      shaker486 - shaker
      left right - hand
      shot80 - shot
      ingredient55 ingredient436 ingredient395 - ingredient
      cocktail1 - cocktail
      dispenser412 dispenser189 dispenser458 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker486)
  (ontable shot80)
  (dispenses dispenser412 ingredient55)
  (dispenses dispenser189 ingredient436)
  (dispenses dispenser458 ingredient395)
  (clean shaker486)
  (clean shot80)
  (empty shaker486)
  (empty shot80)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker486 l0)
  (shaker-level shaker486 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient436)
  (cocktail-part2 cocktail1 ingredient55)
)
 (:goal
  (and
      (contains shot80 cocktail1)
)))
