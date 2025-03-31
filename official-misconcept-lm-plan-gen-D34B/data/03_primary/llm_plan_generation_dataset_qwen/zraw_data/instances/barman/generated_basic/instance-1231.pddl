(define (problem prob)
 (:domain barman)
 (:objects 
      shaker232 - shaker
      left right - hand
      shot429 shot392 shot220 - shot
      ingredient498 ingredient284 ingredient98 ingredient211 - ingredient
      cocktail344 - cocktail
      dispenser328 dispenser456 dispenser210 dispenser414 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker232)
  (ontable shot429)
  (ontable shot392)
  (ontable shot220)
  (dispenses dispenser328 ingredient498)
  (dispenses dispenser456 ingredient284)
  (dispenses dispenser210 ingredient98)
  (dispenses dispenser414 ingredient211)
  (clean shaker232)
  (clean shot429)
  (clean shot392)
  (clean shot220)
  (empty shaker232)
  (empty shot429)
  (empty shot392)
  (empty shot220)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker232 l0)
  (shaker-level shaker232 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail344 ingredient98)
  (cocktail-part2 cocktail344 ingredient211)
)
 (:goal
  (and
      (contains shot429 cocktail344)
      (contains shot392 ingredient211)
)))
