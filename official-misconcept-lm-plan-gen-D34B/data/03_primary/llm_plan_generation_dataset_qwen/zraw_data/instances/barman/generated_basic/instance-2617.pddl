(define (problem prob)
 (:domain barman)
 (:objects 
      shaker409 - shaker
      left right - hand
      shot45 shot61 - shot
      ingredient15 ingredient62 ingredient204 - ingredient
      cocktail218 - cocktail
      dispenser444 dispenser238 dispenser57 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker409)
  (ontable shot45)
  (ontable shot61)
  (dispenses dispenser444 ingredient15)
  (dispenses dispenser238 ingredient62)
  (dispenses dispenser57 ingredient204)
  (clean shaker409)
  (clean shot45)
  (clean shot61)
  (empty shaker409)
  (empty shot45)
  (empty shot61)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker409 l0)
  (shaker-level shaker409 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail218 ingredient62)
  (cocktail-part2 cocktail218 ingredient204)
)
 (:goal
  (and
      (contains shot45 cocktail218)
)))
