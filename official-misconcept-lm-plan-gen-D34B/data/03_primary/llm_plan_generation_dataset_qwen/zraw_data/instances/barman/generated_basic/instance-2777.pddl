(define (problem prob)
 (:domain barman)
 (:objects 
      shaker257 - shaker
      left right - hand
      shot483 - shot
      ingredient361 ingredient169 ingredient101 ingredient182 - ingredient
      cocktail275 - cocktail
      dispenser368 dispenser380 dispenser315 dispenser77 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker257)
  (ontable shot483)
  (dispenses dispenser368 ingredient361)
  (dispenses dispenser380 ingredient169)
  (dispenses dispenser315 ingredient101)
  (dispenses dispenser77 ingredient182)
  (clean shaker257)
  (clean shot483)
  (empty shaker257)
  (empty shot483)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker257 l0)
  (shaker-level shaker257 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail275 ingredient169)
  (cocktail-part2 cocktail275 ingredient361)
)
 (:goal
  (and
      (contains shot483 cocktail275)
)))
