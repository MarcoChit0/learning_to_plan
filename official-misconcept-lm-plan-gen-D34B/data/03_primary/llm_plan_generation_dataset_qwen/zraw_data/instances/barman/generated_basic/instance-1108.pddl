(define (problem prob)
 (:domain barman)
 (:objects 
      shaker486 - shaker
      left right - hand
      shot416 shot394 - shot
      ingredient74 ingredient124 - ingredient
      cocktail1 - cocktail
      dispenser0 dispenser374 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker486)
  (ontable shot416)
  (ontable shot394)
  (dispenses dispenser0 ingredient74)
  (dispenses dispenser374 ingredient124)
  (clean shaker486)
  (clean shot416)
  (clean shot394)
  (empty shaker486)
  (empty shot416)
  (empty shot394)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker486 l0)
  (shaker-level shaker486 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient124)
  (cocktail-part2 cocktail1 ingredient74)
)
 (:goal
  (and
      (contains shot416 cocktail1)
)))
