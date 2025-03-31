(define (problem prob)
 (:domain barman)
 (:objects 
      shaker37 - shaker
      left right - hand
      shot27 shot231 shot285 - shot
      ingredient240 ingredient386 - ingredient
      cocktail242 - cocktail
      dispenser302 dispenser420 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker37)
  (ontable shot27)
  (ontable shot231)
  (ontable shot285)
  (dispenses dispenser302 ingredient240)
  (dispenses dispenser420 ingredient386)
  (clean shaker37)
  (clean shot27)
  (clean shot231)
  (clean shot285)
  (empty shaker37)
  (empty shot27)
  (empty shot231)
  (empty shot285)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker37 l0)
  (shaker-level shaker37 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail242 ingredient386)
  (cocktail-part2 cocktail242 ingredient240)
)
 (:goal
  (and
      (contains shot27 cocktail242)
      (contains shot231 cocktail242)
)))
