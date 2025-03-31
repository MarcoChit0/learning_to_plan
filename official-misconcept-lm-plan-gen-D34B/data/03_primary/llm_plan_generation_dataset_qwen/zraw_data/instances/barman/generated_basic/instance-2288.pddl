(define (problem prob)
 (:domain barman)
 (:objects 
      shaker12 - shaker
      left right - hand
      shot352 - shot
      ingredient307 ingredient324 ingredient32 - ingredient
      cocktail286 - cocktail
      dispenser84 dispenser296 dispenser115 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker12)
  (ontable shot352)
  (dispenses dispenser84 ingredient307)
  (dispenses dispenser296 ingredient324)
  (dispenses dispenser115 ingredient32)
  (clean shaker12)
  (clean shot352)
  (empty shaker12)
  (empty shot352)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker12 l0)
  (shaker-level shaker12 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail286 ingredient32)
  (cocktail-part2 cocktail286 ingredient324)
)
 (:goal
  (and
      (contains shot352 cocktail286)
)))
