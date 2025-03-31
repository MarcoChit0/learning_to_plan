(define (problem prob)
 (:domain barman)
 (:objects 
      shaker85 - shaker
      left right - hand
      shot401 shot276 - shot
      ingredient224 ingredient405 ingredient270 - ingredient
      cocktail245 - cocktail
      dispenser364 dispenser415 dispenser478 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker85)
  (ontable shot401)
  (ontable shot276)
  (dispenses dispenser364 ingredient224)
  (dispenses dispenser415 ingredient405)
  (dispenses dispenser478 ingredient270)
  (clean shaker85)
  (clean shot401)
  (clean shot276)
  (empty shaker85)
  (empty shot401)
  (empty shot276)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker85 l0)
  (shaker-level shaker85 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail245 ingredient224)
  (cocktail-part2 cocktail245 ingredient405)
)
 (:goal
  (and
      (contains shot401 cocktail245)
)))
