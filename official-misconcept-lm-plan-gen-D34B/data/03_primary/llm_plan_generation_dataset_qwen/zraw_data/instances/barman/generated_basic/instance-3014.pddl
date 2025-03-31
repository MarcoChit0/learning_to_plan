(define (problem prob)
 (:domain barman)
 (:objects 
      shaker493 - shaker
      left right - hand
      shot67 shot121 - shot
      ingredient273 ingredient21 ingredient252 - ingredient
      cocktail132 - cocktail
      dispenser339 dispenser16 dispenser440 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker493)
  (ontable shot67)
  (ontable shot121)
  (dispenses dispenser339 ingredient273)
  (dispenses dispenser16 ingredient21)
  (dispenses dispenser440 ingredient252)
  (clean shaker493)
  (clean shot67)
  (clean shot121)
  (empty shaker493)
  (empty shot67)
  (empty shot121)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker493 l0)
  (shaker-level shaker493 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail132 ingredient21)
  (cocktail-part2 cocktail132 ingredient273)
)
 (:goal
  (and
      (contains shot67 cocktail132)
)))
