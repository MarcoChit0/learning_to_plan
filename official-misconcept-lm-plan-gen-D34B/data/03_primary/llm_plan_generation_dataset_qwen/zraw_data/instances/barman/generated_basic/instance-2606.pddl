(define (problem prob)
 (:domain barman)
 (:objects 
      shaker145 - shaker
      left right - hand
      shot188 shot64 - shot
      ingredient280 ingredient309 ingredient11 - ingredient
      cocktail22 - cocktail
      dispenser46 dispenser26 dispenser47 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker145)
  (ontable shot188)
  (ontable shot64)
  (dispenses dispenser46 ingredient280)
  (dispenses dispenser26 ingredient309)
  (dispenses dispenser47 ingredient11)
  (clean shaker145)
  (clean shot188)
  (clean shot64)
  (empty shaker145)
  (empty shot188)
  (empty shot64)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker145 l0)
  (shaker-level shaker145 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail22 ingredient309)
  (cocktail-part2 cocktail22 ingredient11)
)
 (:goal
  (and
      (contains shot188 cocktail22)
)))
