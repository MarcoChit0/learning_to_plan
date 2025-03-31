(define (problem prob)
 (:domain barman)
 (:objects 
      shaker388 - shaker
      left right - hand
      shot245 shot35 - shot
      ingredient310 ingredient242 - ingredient
      cocktail352 - cocktail
      dispenser7 dispenser109 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker388)
  (ontable shot245)
  (ontable shot35)
  (dispenses dispenser7 ingredient310)
  (dispenses dispenser109 ingredient242)
  (clean shaker388)
  (clean shot245)
  (clean shot35)
  (empty shaker388)
  (empty shot245)
  (empty shot35)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker388 l0)
  (shaker-level shaker388 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail352 ingredient310)
  (cocktail-part2 cocktail352 ingredient242)
)
 (:goal
  (and
      (contains shot245 cocktail352)
)))
