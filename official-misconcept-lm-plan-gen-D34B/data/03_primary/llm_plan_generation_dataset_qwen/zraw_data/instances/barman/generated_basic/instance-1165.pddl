(define (problem prob)
 (:domain barman)
 (:objects 
      shaker394 - shaker
      left right - hand
      shot424 shot236 - shot
      ingredient65 ingredient482 ingredient224 - ingredient
      cocktail1 - cocktail
      dispenser170 dispenser90 dispenser134 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker394)
  (ontable shot424)
  (ontable shot236)
  (dispenses dispenser170 ingredient65)
  (dispenses dispenser90 ingredient482)
  (dispenses dispenser134 ingredient224)
  (clean shaker394)
  (clean shot424)
  (clean shot236)
  (empty shaker394)
  (empty shot424)
  (empty shot236)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker394 l0)
  (shaker-level shaker394 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient482)
  (cocktail-part2 cocktail1 ingredient224)
)
 (:goal
  (and
      (contains shot424 cocktail1)
)))
