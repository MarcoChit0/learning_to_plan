(define (problem prob)
 (:domain barman)
 (:objects 
      shaker446 - shaker
      left right - hand
      shot116 - shot
      ingredient151 ingredient269 - ingredient
      cocktail345 - cocktail
      dispenser120 dispenser497 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker446)
  (ontable shot116)
  (dispenses dispenser120 ingredient151)
  (dispenses dispenser497 ingredient269)
  (clean shaker446)
  (clean shot116)
  (empty shaker446)
  (empty shot116)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker446 l0)
  (shaker-level shaker446 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail345 ingredient151)
  (cocktail-part2 cocktail345 ingredient269)
)
 (:goal
  (and
      (contains shot116 cocktail345)
)))
