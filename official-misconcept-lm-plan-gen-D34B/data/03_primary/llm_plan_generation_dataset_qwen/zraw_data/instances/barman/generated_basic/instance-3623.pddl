(define (problem prob)
 (:domain barman)
 (:objects 
      shaker213 - shaker
      left right - hand
      shot154 shot272 shot456 - shot
      ingredient378 ingredient411 ingredient370 - ingredient
      cocktail124 - cocktail
      dispenser389 dispenser265 dispenser192 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker213)
  (ontable shot154)
  (ontable shot272)
  (ontable shot456)
  (dispenses dispenser389 ingredient378)
  (dispenses dispenser265 ingredient411)
  (dispenses dispenser192 ingredient370)
  (clean shaker213)
  (clean shot154)
  (clean shot272)
  (clean shot456)
  (empty shaker213)
  (empty shot154)
  (empty shot272)
  (empty shot456)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker213 l0)
  (shaker-level shaker213 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail124 ingredient411)
  (cocktail-part2 cocktail124 ingredient370)
)
 (:goal
  (and
      (contains shot154 cocktail124)
      (contains shot272 ingredient411)
)))
