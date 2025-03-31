(define (problem prob)
 (:domain barman)
 (:objects 
      shaker499 - shaker
      left right - hand
      shot496 shot7 - shot
      ingredient479 ingredient241 ingredient54 - ingredient
      cocktail1 - cocktail
      dispenser444 dispenser58 dispenser191 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker499)
  (ontable shot496)
  (ontable shot7)
  (dispenses dispenser444 ingredient479)
  (dispenses dispenser58 ingredient241)
  (dispenses dispenser191 ingredient54)
  (clean shaker499)
  (clean shot496)
  (clean shot7)
  (empty shaker499)
  (empty shot496)
  (empty shot7)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker499 l0)
  (shaker-level shaker499 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient479)
  (cocktail-part2 cocktail1 ingredient241)
)
 (:goal
  (and
      (contains shot496 cocktail1)
)))
