(define (problem prob)
 (:domain barman)
 (:objects 
      shaker390 - shaker
      left right - hand
      shot163 shot55 shot452 - shot
      ingredient339 ingredient487 - ingredient
      cocktail333 - cocktail
      dispenser257 dispenser105 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker390)
  (ontable shot163)
  (ontable shot55)
  (ontable shot452)
  (dispenses dispenser257 ingredient339)
  (dispenses dispenser105 ingredient487)
  (clean shaker390)
  (clean shot163)
  (clean shot55)
  (clean shot452)
  (empty shaker390)
  (empty shot163)
  (empty shot55)
  (empty shot452)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker390 l0)
  (shaker-level shaker390 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail333 ingredient339)
  (cocktail-part2 cocktail333 ingredient487)
)
 (:goal
  (and
      (contains shot163 cocktail333)
      (contains shot55 cocktail333)
)))
