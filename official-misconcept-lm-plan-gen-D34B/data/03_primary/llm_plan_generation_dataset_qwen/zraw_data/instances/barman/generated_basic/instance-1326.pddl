(define (problem prob)
 (:domain barman)
 (:objects 
      shaker27 - shaker
      left right - hand
      shot345 - shot
      ingredient439 ingredient493 ingredient226 ingredient33 - ingredient
      cocktail97 - cocktail
      dispenser378 dispenser444 dispenser348 dispenser445 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker27)
  (ontable shot345)
  (dispenses dispenser378 ingredient439)
  (dispenses dispenser444 ingredient493)
  (dispenses dispenser348 ingredient226)
  (dispenses dispenser445 ingredient33)
  (clean shaker27)
  (clean shot345)
  (empty shaker27)
  (empty shot345)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker27 l0)
  (shaker-level shaker27 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail97 ingredient33)
  (cocktail-part2 cocktail97 ingredient439)
)
 (:goal
  (and
      (contains shot345 cocktail97)
)))
