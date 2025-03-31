(define (problem prob)
 (:domain barman)
 (:objects 
      shaker228 - shaker
      left right - hand
      shot156 shot196 shot52 - shot
      ingredient460 ingredient214 ingredient498 ingredient309 - ingredient
      cocktail1 - cocktail
      dispenser322 dispenser304 dispenser109 dispenser70 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker228)
  (ontable shot156)
  (ontable shot196)
  (ontable shot52)
  (dispenses dispenser322 ingredient460)
  (dispenses dispenser304 ingredient214)
  (dispenses dispenser109 ingredient498)
  (dispenses dispenser70 ingredient309)
  (clean shaker228)
  (clean shot156)
  (clean shot196)
  (clean shot52)
  (empty shaker228)
  (empty shot156)
  (empty shot196)
  (empty shot52)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker228 l0)
  (shaker-level shaker228 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient498)
  (cocktail-part2 cocktail1 ingredient460)
)
 (:goal
  (and
      (contains shot156 cocktail1)
      (contains shot196 cocktail1)
)))
