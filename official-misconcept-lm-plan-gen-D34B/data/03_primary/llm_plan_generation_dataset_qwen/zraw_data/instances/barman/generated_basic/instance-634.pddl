(define (problem prob)
 (:domain barman)
 (:objects 
      shaker37 - shaker
      left right - hand
      shot461 shot404 shot27 - shot
      ingredient44 ingredient316 ingredient475 - ingredient
      cocktail1 - cocktail
      dispenser271 dispenser65 dispenser315 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker37)
  (ontable shot461)
  (ontable shot404)
  (ontable shot27)
  (dispenses dispenser271 ingredient44)
  (dispenses dispenser65 ingredient316)
  (dispenses dispenser315 ingredient475)
  (clean shaker37)
  (clean shot461)
  (clean shot404)
  (clean shot27)
  (empty shaker37)
  (empty shot461)
  (empty shot404)
  (empty shot27)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker37 l0)
  (shaker-level shaker37 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient316)
  (cocktail-part2 cocktail1 ingredient44)
)
 (:goal
  (and
      (contains shot461 cocktail1)
      (contains shot404 ingredient44)
)))
