(define (problem prob)
 (:domain barman)
 (:objects 
      shaker493 - shaker
      left right - hand
      shot257 shot28 - shot
      ingredient46 ingredient388 ingredient110 - ingredient
      cocktail35 - cocktail
      dispenser23 dispenser498 dispenser496 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker493)
  (ontable shot257)
  (ontable shot28)
  (dispenses dispenser23 ingredient46)
  (dispenses dispenser498 ingredient388)
  (dispenses dispenser496 ingredient110)
  (clean shaker493)
  (clean shot257)
  (clean shot28)
  (empty shaker493)
  (empty shot257)
  (empty shot28)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker493 l0)
  (shaker-level shaker493 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail35 ingredient388)
  (cocktail-part2 cocktail35 ingredient46)
)
 (:goal
  (and
      (contains shot257 cocktail35)
)))
