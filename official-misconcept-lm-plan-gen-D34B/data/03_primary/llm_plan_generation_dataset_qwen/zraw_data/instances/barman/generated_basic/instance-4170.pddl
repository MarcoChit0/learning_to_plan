(define (problem prob)
 (:domain barman)
 (:objects 
      shaker315 - shaker
      left right - hand
      shot432 shot386 - shot
      ingredient8 ingredient31 - ingredient
      cocktail102 - cocktail
      dispenser230 dispenser83 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker315)
  (ontable shot432)
  (ontable shot386)
  (dispenses dispenser230 ingredient8)
  (dispenses dispenser83 ingredient31)
  (clean shaker315)
  (clean shot432)
  (clean shot386)
  (empty shaker315)
  (empty shot432)
  (empty shot386)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker315 l0)
  (shaker-level shaker315 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail102 ingredient8)
  (cocktail-part2 cocktail102 ingredient31)
)
 (:goal
  (and
      (contains shot432 cocktail102)
)))
