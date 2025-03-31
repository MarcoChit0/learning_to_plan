(define (problem prob)
 (:domain barman)
 (:objects 
      shaker68 - shaker
      left right - hand
      shot330 - shot
      ingredient255 ingredient132 - ingredient
      cocktail32 - cocktail
      dispenser440 dispenser231 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker68)
  (ontable shot330)
  (dispenses dispenser440 ingredient255)
  (dispenses dispenser231 ingredient132)
  (clean shaker68)
  (clean shot330)
  (empty shaker68)
  (empty shot330)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker68 l0)
  (shaker-level shaker68 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail32 ingredient255)
  (cocktail-part2 cocktail32 ingredient132)
)
 (:goal
  (and
      (contains shot330 cocktail32)
)))
