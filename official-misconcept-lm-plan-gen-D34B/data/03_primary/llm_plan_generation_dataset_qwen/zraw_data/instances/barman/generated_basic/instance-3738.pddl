(define (problem prob)
 (:domain barman)
 (:objects 
      shaker120 - shaker
      left right - hand
      shot194 shot422 shot271 - shot
      ingredient59 ingredient294 ingredient207 - ingredient
      cocktail231 - cocktail
      dispenser257 dispenser440 dispenser421 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker120)
  (ontable shot194)
  (ontable shot422)
  (ontable shot271)
  (dispenses dispenser257 ingredient59)
  (dispenses dispenser440 ingredient294)
  (dispenses dispenser421 ingredient207)
  (clean shaker120)
  (clean shot194)
  (clean shot422)
  (clean shot271)
  (empty shaker120)
  (empty shot194)
  (empty shot422)
  (empty shot271)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker120 l0)
  (shaker-level shaker120 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail231 ingredient294)
  (cocktail-part2 cocktail231 ingredient207)
)
 (:goal
  (and
      (contains shot194 cocktail231)
      (contains shot422 cocktail231)
)))
