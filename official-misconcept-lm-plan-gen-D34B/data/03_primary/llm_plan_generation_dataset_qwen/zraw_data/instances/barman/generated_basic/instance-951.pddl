(define (problem prob)
 (:domain barman)
 (:objects 
      shaker21 - shaker
      left right - hand
      shot156 - shot
      ingredient410 ingredient76 ingredient229 ingredient223 - ingredient
      cocktail1 - cocktail
      dispenser50 dispenser445 dispenser55 dispenser461 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker21)
  (ontable shot156)
  (dispenses dispenser50 ingredient410)
  (dispenses dispenser445 ingredient76)
  (dispenses dispenser55 ingredient229)
  (dispenses dispenser461 ingredient223)
  (clean shaker21)
  (clean shot156)
  (empty shaker21)
  (empty shot156)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker21 l0)
  (shaker-level shaker21 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient223)
  (cocktail-part2 cocktail1 ingredient76)
)
 (:goal
  (and
      (contains shot156 cocktail1)
)))
