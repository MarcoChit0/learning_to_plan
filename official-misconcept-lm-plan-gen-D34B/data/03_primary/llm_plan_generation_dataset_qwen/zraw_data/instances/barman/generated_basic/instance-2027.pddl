(define (problem prob)
 (:domain barman)
 (:objects 
      shaker381 - shaker
      left right - hand
      shot41 shot360 - shot
      ingredient459 ingredient183 - ingredient
      cocktail306 - cocktail
      dispenser212 dispenser381 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker381)
  (ontable shot41)
  (ontable shot360)
  (dispenses dispenser212 ingredient459)
  (dispenses dispenser381 ingredient183)
  (clean shaker381)
  (clean shot41)
  (clean shot360)
  (empty shaker381)
  (empty shot41)
  (empty shot360)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker381 l0)
  (shaker-level shaker381 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail306 ingredient183)
  (cocktail-part2 cocktail306 ingredient459)
)
 (:goal
  (and
      (contains shot41 cocktail306)
)))
