(define (problem prob)
 (:domain barman)
 (:objects 
      shaker99 - shaker
      left right - hand
      shot315 shot352 shot98 - shot
      ingredient230 ingredient464 ingredient138 - ingredient
      cocktail313 - cocktail
      dispenser194 dispenser134 dispenser311 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker99)
  (ontable shot315)
  (ontable shot352)
  (ontable shot98)
  (dispenses dispenser194 ingredient230)
  (dispenses dispenser134 ingredient464)
  (dispenses dispenser311 ingredient138)
  (clean shaker99)
  (clean shot315)
  (clean shot352)
  (clean shot98)
  (empty shaker99)
  (empty shot315)
  (empty shot352)
  (empty shot98)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker99 l0)
  (shaker-level shaker99 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail313 ingredient464)
  (cocktail-part2 cocktail313 ingredient230)
)
 (:goal
  (and
      (contains shot315 cocktail313)
      (contains shot352 cocktail313)
)))
