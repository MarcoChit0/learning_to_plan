(define (problem prob)
 (:domain barman)
 (:objects 
      shaker21 - shaker
      left right - hand
      shot318 - shot
      ingredient460 ingredient360 ingredient380 ingredient268 - ingredient
      cocktail182 - cocktail
      dispenser30 dispenser499 dispenser212 dispenser315 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker21)
  (ontable shot318)
  (dispenses dispenser30 ingredient460)
  (dispenses dispenser499 ingredient360)
  (dispenses dispenser212 ingredient380)
  (dispenses dispenser315 ingredient268)
  (clean shaker21)
  (clean shot318)
  (empty shaker21)
  (empty shot318)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker21 l0)
  (shaker-level shaker21 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail182 ingredient360)
  (cocktail-part2 cocktail182 ingredient268)
)
 (:goal
  (and
      (contains shot318 cocktail182)
)))
