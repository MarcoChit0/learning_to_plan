(define (problem prob)
 (:domain barman)
 (:objects 
      shaker54 - shaker
      left right - hand
      shot494 shot142 - shot
      ingredient408 ingredient411 ingredient110 ingredient26 - ingredient
      cocktail324 - cocktail
      dispenser60 dispenser452 dispenser12 dispenser358 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker54)
  (ontable shot494)
  (ontable shot142)
  (dispenses dispenser60 ingredient408)
  (dispenses dispenser452 ingredient411)
  (dispenses dispenser12 ingredient110)
  (dispenses dispenser358 ingredient26)
  (clean shaker54)
  (clean shot494)
  (clean shot142)
  (empty shaker54)
  (empty shot494)
  (empty shot142)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker54 l0)
  (shaker-level shaker54 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail324 ingredient26)
  (cocktail-part2 cocktail324 ingredient408)
)
 (:goal
  (and
      (contains shot494 cocktail324)
)))
