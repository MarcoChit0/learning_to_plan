(define (problem prob)
 (:domain barman)
 (:objects 
      shaker153 - shaker
      left right - hand
      shot416 shot11 - shot
      ingredient442 ingredient389 ingredient70 - ingredient
      cocktail308 - cocktail
      dispenser166 dispenser145 dispenser35 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker153)
  (ontable shot416)
  (ontable shot11)
  (dispenses dispenser166 ingredient442)
  (dispenses dispenser145 ingredient389)
  (dispenses dispenser35 ingredient70)
  (clean shaker153)
  (clean shot416)
  (clean shot11)
  (empty shaker153)
  (empty shot416)
  (empty shot11)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker153 l0)
  (shaker-level shaker153 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail308 ingredient442)
  (cocktail-part2 cocktail308 ingredient389)
)
 (:goal
  (and
      (contains shot416 cocktail308)
)))
