(define (problem prob)
 (:domain barman)
 (:objects 
      shaker200 - shaker
      left right - hand
      shot134 shot407 - shot
      ingredient382 ingredient232 ingredient179 ingredient439 - ingredient
      cocktail9 - cocktail
      dispenser356 dispenser103 dispenser216 dispenser149 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker200)
  (ontable shot134)
  (ontable shot407)
  (dispenses dispenser356 ingredient382)
  (dispenses dispenser103 ingredient232)
  (dispenses dispenser216 ingredient179)
  (dispenses dispenser149 ingredient439)
  (clean shaker200)
  (clean shot134)
  (clean shot407)
  (empty shaker200)
  (empty shot134)
  (empty shot407)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker200 l0)
  (shaker-level shaker200 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail9 ingredient439)
  (cocktail-part2 cocktail9 ingredient232)
)
 (:goal
  (and
      (contains shot134 cocktail9)
)))
