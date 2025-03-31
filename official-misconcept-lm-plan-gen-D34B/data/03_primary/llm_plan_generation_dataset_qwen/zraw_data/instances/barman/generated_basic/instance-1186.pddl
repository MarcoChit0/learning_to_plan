(define (problem prob)
 (:domain barman)
 (:objects 
      shaker36 - shaker
      left right - hand
      shot54 shot302 - shot
      ingredient302 ingredient181 - ingredient
      cocktail293 - cocktail
      dispenser170 dispenser65 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker36)
  (ontable shot54)
  (ontable shot302)
  (dispenses dispenser170 ingredient302)
  (dispenses dispenser65 ingredient181)
  (clean shaker36)
  (clean shot54)
  (clean shot302)
  (empty shaker36)
  (empty shot54)
  (empty shot302)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker36 l0)
  (shaker-level shaker36 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail293 ingredient302)
  (cocktail-part2 cocktail293 ingredient181)
)
 (:goal
  (and
      (contains shot54 cocktail293)
)))
