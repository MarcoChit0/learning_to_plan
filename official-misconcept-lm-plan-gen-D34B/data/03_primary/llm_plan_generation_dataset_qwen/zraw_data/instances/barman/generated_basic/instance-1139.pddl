(define (problem prob)
 (:domain barman)
 (:objects 
      shaker275 - shaker
      left right - hand
      shot54 shot267 - shot
      ingredient275 ingredient304 ingredient176 - ingredient
      cocktail1 - cocktail
      dispenser252 dispenser250 dispenser324 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker275)
  (ontable shot54)
  (ontable shot267)
  (dispenses dispenser252 ingredient275)
  (dispenses dispenser250 ingredient304)
  (dispenses dispenser324 ingredient176)
  (clean shaker275)
  (clean shot54)
  (clean shot267)
  (empty shaker275)
  (empty shot54)
  (empty shot267)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker275 l0)
  (shaker-level shaker275 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient275)
  (cocktail-part2 cocktail1 ingredient304)
)
 (:goal
  (and
      (contains shot54 cocktail1)
)))
