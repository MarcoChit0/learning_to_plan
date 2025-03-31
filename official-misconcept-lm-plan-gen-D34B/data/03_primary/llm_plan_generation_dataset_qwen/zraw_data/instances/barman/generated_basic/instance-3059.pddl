(define (problem prob)
 (:domain barman)
 (:objects 
      shaker316 - shaker
      left right - hand
      shot427 shot312 shot51 shot44 - shot
      ingredient51 ingredient328 ingredient200 ingredient55 - ingredient
      cocktail63 - cocktail
      dispenser330 dispenser124 dispenser199 dispenser162 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker316)
  (ontable shot427)
  (ontable shot312)
  (ontable shot51)
  (ontable shot44)
  (dispenses dispenser330 ingredient51)
  (dispenses dispenser124 ingredient328)
  (dispenses dispenser199 ingredient200)
  (dispenses dispenser162 ingredient55)
  (clean shaker316)
  (clean shot427)
  (clean shot312)
  (clean shot51)
  (clean shot44)
  (empty shaker316)
  (empty shot427)
  (empty shot312)
  (empty shot51)
  (empty shot44)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker316 l0)
  (shaker-level shaker316 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail63 ingredient55)
  (cocktail-part2 cocktail63 ingredient328)
)
 (:goal
  (and
      (contains shot427 cocktail63)
      (contains shot312 ingredient328)
      (contains shot51 cocktail63)
)))
