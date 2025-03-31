(define (problem prob)
 (:domain barman)
 (:objects 
      shaker223 - shaker
      left right - hand
      shot369 shot461 - shot
      ingredient172 ingredient80 ingredient193 - ingredient
      cocktail159 - cocktail
      dispenser10 dispenser16 dispenser37 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker223)
  (ontable shot369)
  (ontable shot461)
  (dispenses dispenser10 ingredient172)
  (dispenses dispenser16 ingredient80)
  (dispenses dispenser37 ingredient193)
  (clean shaker223)
  (clean shot369)
  (clean shot461)
  (empty shaker223)
  (empty shot369)
  (empty shot461)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker223 l0)
  (shaker-level shaker223 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail159 ingredient80)
  (cocktail-part2 cocktail159 ingredient193)
)
 (:goal
  (and
      (contains shot369 cocktail159)
)))
