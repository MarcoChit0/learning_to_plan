(define (problem prob)
 (:domain barman)
 (:objects 
      shaker246 - shaker
      left right - hand
      shot285 shot357 shot266 - shot
      ingredient44 ingredient98 - ingredient
      cocktail97 - cocktail
      dispenser390 dispenser492 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker246)
  (ontable shot285)
  (ontable shot357)
  (ontable shot266)
  (dispenses dispenser390 ingredient44)
  (dispenses dispenser492 ingredient98)
  (clean shaker246)
  (clean shot285)
  (clean shot357)
  (clean shot266)
  (empty shaker246)
  (empty shot285)
  (empty shot357)
  (empty shot266)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker246 l0)
  (shaker-level shaker246 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail97 ingredient98)
  (cocktail-part2 cocktail97 ingredient44)
)
 (:goal
  (and
      (contains shot285 cocktail97)
      (contains shot357 cocktail97)
)))
