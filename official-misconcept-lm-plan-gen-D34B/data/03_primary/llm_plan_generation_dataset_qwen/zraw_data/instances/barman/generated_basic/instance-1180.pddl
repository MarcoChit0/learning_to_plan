(define (problem prob)
 (:domain barman)
 (:objects 
      shaker80 - shaker
      left right - hand
      shot464 shot280 - shot
      ingredient150 ingredient323 ingredient105 - ingredient
      cocktail1 - cocktail
      dispenser231 dispenser404 dispenser0 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker80)
  (ontable shot464)
  (ontable shot280)
  (dispenses dispenser231 ingredient150)
  (dispenses dispenser404 ingredient323)
  (dispenses dispenser0 ingredient105)
  (clean shaker80)
  (clean shot464)
  (clean shot280)
  (empty shaker80)
  (empty shot464)
  (empty shot280)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker80 l0)
  (shaker-level shaker80 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient323)
  (cocktail-part2 cocktail1 ingredient150)
)
 (:goal
  (and
      (contains shot464 cocktail1)
)))
