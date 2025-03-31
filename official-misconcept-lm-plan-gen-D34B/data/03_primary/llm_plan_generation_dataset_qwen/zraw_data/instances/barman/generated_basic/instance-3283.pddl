(define (problem prob)
 (:domain barman)
 (:objects 
      shaker49 - shaker
      left right - hand
      shot202 shot479 - shot
      ingredient466 ingredient100 ingredient453 - ingredient
      cocktail408 - cocktail
      dispenser255 dispenser142 dispenser330 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker49)
  (ontable shot202)
  (ontable shot479)
  (dispenses dispenser255 ingredient466)
  (dispenses dispenser142 ingredient100)
  (dispenses dispenser330 ingredient453)
  (clean shaker49)
  (clean shot202)
  (clean shot479)
  (empty shaker49)
  (empty shot202)
  (empty shot479)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker49 l0)
  (shaker-level shaker49 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail408 ingredient466)
  (cocktail-part2 cocktail408 ingredient453)
)
 (:goal
  (and
      (contains shot202 cocktail408)
)))
