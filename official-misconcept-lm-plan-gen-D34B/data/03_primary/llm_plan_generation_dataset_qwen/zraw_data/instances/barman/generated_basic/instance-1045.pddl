(define (problem prob)
 (:domain barman)
 (:objects 
      shaker475 - shaker
      left right - hand
      shot113 shot286 shot84 - shot
      ingredient335 ingredient49 ingredient137 ingredient71 - ingredient
      cocktail1 - cocktail
      dispenser34 dispenser467 dispenser57 dispenser210 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker475)
  (ontable shot113)
  (ontable shot286)
  (ontable shot84)
  (dispenses dispenser34 ingredient335)
  (dispenses dispenser467 ingredient49)
  (dispenses dispenser57 ingredient137)
  (dispenses dispenser210 ingredient71)
  (clean shaker475)
  (clean shot113)
  (clean shot286)
  (clean shot84)
  (empty shaker475)
  (empty shot113)
  (empty shot286)
  (empty shot84)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker475 l0)
  (shaker-level shaker475 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient335)
  (cocktail-part2 cocktail1 ingredient71)
)
 (:goal
  (and
      (contains shot113 cocktail1)
      (contains shot286 cocktail1)
)))
