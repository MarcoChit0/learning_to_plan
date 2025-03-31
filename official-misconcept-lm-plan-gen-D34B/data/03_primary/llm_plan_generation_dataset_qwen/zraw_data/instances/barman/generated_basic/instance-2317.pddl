(define (problem prob)
 (:domain barman)
 (:objects 
      shaker73 - shaker
      left right - hand
      shot310 - shot
      ingredient216 ingredient209 - ingredient
      cocktail58 - cocktail
      dispenser177 dispenser308 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker73)
  (ontable shot310)
  (dispenses dispenser177 ingredient216)
  (dispenses dispenser308 ingredient209)
  (clean shaker73)
  (clean shot310)
  (empty shaker73)
  (empty shot310)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker73 l0)
  (shaker-level shaker73 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail58 ingredient216)
  (cocktail-part2 cocktail58 ingredient209)
)
 (:goal
  (and
      (contains shot310 cocktail58)
)))
