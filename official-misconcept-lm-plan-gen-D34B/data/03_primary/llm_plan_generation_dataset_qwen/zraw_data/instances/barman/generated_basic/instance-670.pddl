(define (problem prob)
 (:domain barman)
 (:objects 
      shaker277 - shaker
      left right - hand
      shot398 shot46 - shot
      ingredient484 ingredient476 - ingredient
      cocktail1 - cocktail
      dispenser297 dispenser484 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker277)
  (ontable shot398)
  (ontable shot46)
  (dispenses dispenser297 ingredient484)
  (dispenses dispenser484 ingredient476)
  (clean shaker277)
  (clean shot398)
  (clean shot46)
  (empty shaker277)
  (empty shot398)
  (empty shot46)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker277 l0)
  (shaker-level shaker277 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient476)
  (cocktail-part2 cocktail1 ingredient484)
)
 (:goal
  (and
      (contains shot398 cocktail1)
)))
