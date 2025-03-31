(define (problem prob)
 (:domain barman)
 (:objects 
      shaker113 - shaker
      left right - hand
      shot391 shot109 shot224 - shot
      ingredient156 ingredient483 ingredient246 ingredient436 - ingredient
      cocktail330 - cocktail
      dispenser217 dispenser262 dispenser476 dispenser341 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker113)
  (ontable shot391)
  (ontable shot109)
  (ontable shot224)
  (dispenses dispenser217 ingredient156)
  (dispenses dispenser262 ingredient483)
  (dispenses dispenser476 ingredient246)
  (dispenses dispenser341 ingredient436)
  (clean shaker113)
  (clean shot391)
  (clean shot109)
  (clean shot224)
  (empty shaker113)
  (empty shot391)
  (empty shot109)
  (empty shot224)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker113 l0)
  (shaker-level shaker113 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail330 ingredient246)
  (cocktail-part2 cocktail330 ingredient483)
)
 (:goal
  (and
      (contains shot391 cocktail330)
      (contains shot109 cocktail330)
)))
