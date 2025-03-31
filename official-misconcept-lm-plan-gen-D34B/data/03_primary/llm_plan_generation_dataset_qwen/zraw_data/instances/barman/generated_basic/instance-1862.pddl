(define (problem prob)
 (:domain barman)
 (:objects 
      shaker104 - shaker
      left right - hand
      shot408 shot318 - shot
      ingredient130 ingredient246 - ingredient
      cocktail269 - cocktail
      dispenser33 dispenser316 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker104)
  (ontable shot408)
  (ontable shot318)
  (dispenses dispenser33 ingredient130)
  (dispenses dispenser316 ingredient246)
  (clean shaker104)
  (clean shot408)
  (clean shot318)
  (empty shaker104)
  (empty shot408)
  (empty shot318)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker104 l0)
  (shaker-level shaker104 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail269 ingredient130)
  (cocktail-part2 cocktail269 ingredient246)
)
 (:goal
  (and
      (contains shot408 cocktail269)
)))
