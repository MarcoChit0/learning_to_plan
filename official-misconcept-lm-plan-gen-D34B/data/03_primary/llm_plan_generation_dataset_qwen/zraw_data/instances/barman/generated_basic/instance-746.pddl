(define (problem prob)
 (:domain barman)
 (:objects 
      shaker265 - shaker
      left right - hand
      shot279 shot88 - shot
      ingredient241 ingredient113 ingredient120 - ingredient
      cocktail1 - cocktail
      dispenser128 dispenser254 dispenser130 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker265)
  (ontable shot279)
  (ontable shot88)
  (dispenses dispenser128 ingredient241)
  (dispenses dispenser254 ingredient113)
  (dispenses dispenser130 ingredient120)
  (clean shaker265)
  (clean shot279)
  (clean shot88)
  (empty shaker265)
  (empty shot279)
  (empty shot88)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker265 l0)
  (shaker-level shaker265 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient241)
  (cocktail-part2 cocktail1 ingredient113)
)
 (:goal
  (and
      (contains shot279 cocktail1)
)))
