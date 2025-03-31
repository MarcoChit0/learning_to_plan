(define (problem prob)
 (:domain barman)
 (:objects 
      shaker179 - shaker
      left right - hand
      shot374 shot130 - shot
      ingredient260 ingredient213 ingredient224 ingredient452 - ingredient
      cocktail487 - cocktail
      dispenser167 dispenser113 dispenser311 dispenser380 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker179)
  (ontable shot374)
  (ontable shot130)
  (dispenses dispenser167 ingredient260)
  (dispenses dispenser113 ingredient213)
  (dispenses dispenser311 ingredient224)
  (dispenses dispenser380 ingredient452)
  (clean shaker179)
  (clean shot374)
  (clean shot130)
  (empty shaker179)
  (empty shot374)
  (empty shot130)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker179 l0)
  (shaker-level shaker179 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail487 ingredient224)
  (cocktail-part2 cocktail487 ingredient452)
)
 (:goal
  (and
      (contains shot374 cocktail487)
)))
