(define (problem prob)
 (:domain barman)
 (:objects 
      shaker440 - shaker
      left right - hand
      shot21 shot312 - shot
      ingredient263 ingredient484 ingredient46 - ingredient
      cocktail352 - cocktail
      dispenser238 dispenser305 dispenser293 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker440)
  (ontable shot21)
  (ontable shot312)
  (dispenses dispenser238 ingredient263)
  (dispenses dispenser305 ingredient484)
  (dispenses dispenser293 ingredient46)
  (clean shaker440)
  (clean shot21)
  (clean shot312)
  (empty shaker440)
  (empty shot21)
  (empty shot312)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker440 l0)
  (shaker-level shaker440 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail352 ingredient263)
  (cocktail-part2 cocktail352 ingredient46)
)
 (:goal
  (and
      (contains shot21 cocktail352)
)))
