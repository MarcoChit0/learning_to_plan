(define (problem prob)
 (:domain barman)
 (:objects 
      shaker404 - shaker
      left right - hand
      shot43 shot45 shot458 - shot
      ingredient255 ingredient20 - ingredient
      cocktail246 - cocktail
      dispenser377 dispenser323 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker404)
  (ontable shot43)
  (ontable shot45)
  (ontable shot458)
  (dispenses dispenser377 ingredient255)
  (dispenses dispenser323 ingredient20)
  (clean shaker404)
  (clean shot43)
  (clean shot45)
  (clean shot458)
  (empty shaker404)
  (empty shot43)
  (empty shot45)
  (empty shot458)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker404 l0)
  (shaker-level shaker404 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail246 ingredient20)
  (cocktail-part2 cocktail246 ingredient255)
)
 (:goal
  (and
      (contains shot43 cocktail246)
      (contains shot45 ingredient255)
)))
