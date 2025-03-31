(define (problem prob)
 (:domain barman)
 (:objects 
      shaker479 - shaker
      left right - hand
      shot360 shot430 - shot
      ingredient19 ingredient390 ingredient45 ingredient12 - ingredient
      cocktail166 - cocktail
      dispenser187 dispenser436 dispenser382 dispenser230 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker479)
  (ontable shot360)
  (ontable shot430)
  (dispenses dispenser187 ingredient19)
  (dispenses dispenser436 ingredient390)
  (dispenses dispenser382 ingredient45)
  (dispenses dispenser230 ingredient12)
  (clean shaker479)
  (clean shot360)
  (clean shot430)
  (empty shaker479)
  (empty shot360)
  (empty shot430)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker479 l0)
  (shaker-level shaker479 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail166 ingredient390)
  (cocktail-part2 cocktail166 ingredient19)
)
 (:goal
  (and
      (contains shot360 cocktail166)
)))
