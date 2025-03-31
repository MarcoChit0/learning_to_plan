(define (problem prob)
 (:domain barman)
 (:objects 
      shaker151 - shaker
      left right - hand
      shot176 shot121 shot312 - shot
      ingredient206 ingredient347 - ingredient
      cocktail431 - cocktail
      dispenser287 dispenser418 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker151)
  (ontable shot176)
  (ontable shot121)
  (ontable shot312)
  (dispenses dispenser287 ingredient206)
  (dispenses dispenser418 ingredient347)
  (clean shaker151)
  (clean shot176)
  (clean shot121)
  (clean shot312)
  (empty shaker151)
  (empty shot176)
  (empty shot121)
  (empty shot312)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker151 l0)
  (shaker-level shaker151 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail431 ingredient347)
  (cocktail-part2 cocktail431 ingredient206)
)
 (:goal
  (and
      (contains shot176 cocktail431)
      (contains shot121 ingredient347)
)))
