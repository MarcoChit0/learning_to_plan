(define (problem prob)
 (:domain barman)
 (:objects 
      shaker162 - shaker
      left right - hand
      shot480 shot206 - shot
      ingredient187 ingredient336 ingredient311 - ingredient
      cocktail98 - cocktail
      dispenser186 dispenser123 dispenser422 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker162)
  (ontable shot480)
  (ontable shot206)
  (dispenses dispenser186 ingredient187)
  (dispenses dispenser123 ingredient336)
  (dispenses dispenser422 ingredient311)
  (clean shaker162)
  (clean shot480)
  (clean shot206)
  (empty shaker162)
  (empty shot480)
  (empty shot206)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker162 l0)
  (shaker-level shaker162 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail98 ingredient187)
  (cocktail-part2 cocktail98 ingredient311)
)
 (:goal
  (and
      (contains shot480 cocktail98)
)))
