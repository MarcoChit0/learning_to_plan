(define (problem prob)
 (:domain barman)
 (:objects 
      shaker304 - shaker
      left right - hand
      shot134 shot49 shot283 - shot
      ingredient494 ingredient175 - ingredient
      cocktail91 - cocktail
      dispenser107 dispenser432 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker304)
  (ontable shot134)
  (ontable shot49)
  (ontable shot283)
  (dispenses dispenser107 ingredient494)
  (dispenses dispenser432 ingredient175)
  (clean shaker304)
  (clean shot134)
  (clean shot49)
  (clean shot283)
  (empty shaker304)
  (empty shot134)
  (empty shot49)
  (empty shot283)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker304 l0)
  (shaker-level shaker304 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail91 ingredient494)
  (cocktail-part2 cocktail91 ingredient175)
)
 (:goal
  (and
      (contains shot134 cocktail91)
      (contains shot49 cocktail91)
)))
