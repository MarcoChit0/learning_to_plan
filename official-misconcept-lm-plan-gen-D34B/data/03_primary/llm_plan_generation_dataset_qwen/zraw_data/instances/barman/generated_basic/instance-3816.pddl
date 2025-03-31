(define (problem prob)
 (:domain barman)
 (:objects 
      shaker154 - shaker
      left right - hand
      shot31 shot342 - shot
      ingredient191 ingredient397 ingredient12 - ingredient
      cocktail101 - cocktail
      dispenser41 dispenser44 dispenser33 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker154)
  (ontable shot31)
  (ontable shot342)
  (dispenses dispenser41 ingredient191)
  (dispenses dispenser44 ingredient397)
  (dispenses dispenser33 ingredient12)
  (clean shaker154)
  (clean shot31)
  (clean shot342)
  (empty shaker154)
  (empty shot31)
  (empty shot342)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker154 l0)
  (shaker-level shaker154 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail101 ingredient12)
  (cocktail-part2 cocktail101 ingredient191)
)
 (:goal
  (and
      (contains shot31 cocktail101)
)))
