(define (problem prob)
 (:domain barman)
 (:objects 
      shaker192 - shaker
      left right - hand
      shot60 - shot
      ingredient375 ingredient487 - ingredient
      cocktail1 - cocktail
      dispenser74 dispenser368 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker192)
  (ontable shot60)
  (dispenses dispenser74 ingredient375)
  (dispenses dispenser368 ingredient487)
  (clean shaker192)
  (clean shot60)
  (empty shaker192)
  (empty shot60)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker192 l0)
  (shaker-level shaker192 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient487)
  (cocktail-part2 cocktail1 ingredient375)
)
 (:goal
  (and
      (contains shot60 cocktail1)
)))
