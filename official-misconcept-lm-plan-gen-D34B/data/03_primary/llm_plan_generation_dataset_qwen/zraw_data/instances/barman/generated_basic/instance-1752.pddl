(define (problem prob)
 (:domain barman)
 (:objects 
      shaker22 - shaker
      left right - hand
      shot167 shot342 - shot
      ingredient302 ingredient482 ingredient29 - ingredient
      cocktail131 - cocktail
      dispenser55 dispenser229 dispenser218 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker22)
  (ontable shot167)
  (ontable shot342)
  (dispenses dispenser55 ingredient302)
  (dispenses dispenser229 ingredient482)
  (dispenses dispenser218 ingredient29)
  (clean shaker22)
  (clean shot167)
  (clean shot342)
  (empty shaker22)
  (empty shot167)
  (empty shot342)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker22 l0)
  (shaker-level shaker22 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail131 ingredient482)
  (cocktail-part2 cocktail131 ingredient302)
)
 (:goal
  (and
      (contains shot167 cocktail131)
)))
