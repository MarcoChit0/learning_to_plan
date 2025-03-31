(define (problem prob)
 (:domain barman)
 (:objects 
      shaker105 - shaker
      left right - hand
      shot348 shot353 - shot
      ingredient39 ingredient19 ingredient426 ingredient403 - ingredient
      cocktail270 - cocktail
      dispenser350 dispenser298 dispenser232 dispenser182 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker105)
  (ontable shot348)
  (ontable shot353)
  (dispenses dispenser350 ingredient39)
  (dispenses dispenser298 ingredient19)
  (dispenses dispenser232 ingredient426)
  (dispenses dispenser182 ingredient403)
  (clean shaker105)
  (clean shot348)
  (clean shot353)
  (empty shaker105)
  (empty shot348)
  (empty shot353)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker105 l0)
  (shaker-level shaker105 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail270 ingredient403)
  (cocktail-part2 cocktail270 ingredient19)
)
 (:goal
  (and
      (contains shot348 cocktail270)
)))
