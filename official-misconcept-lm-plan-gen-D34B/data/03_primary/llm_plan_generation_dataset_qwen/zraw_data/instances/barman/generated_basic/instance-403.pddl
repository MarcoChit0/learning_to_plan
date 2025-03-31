(define (problem prob)
 (:domain barman)
 (:objects 
      shaker473 - shaker
      left right - hand
      shot170 shot283 - shot
      ingredient10 ingredient133 - ingredient
      cocktail1 - cocktail
      dispenser34 dispenser212 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker473)
  (ontable shot170)
  (ontable shot283)
  (dispenses dispenser34 ingredient10)
  (dispenses dispenser212 ingredient133)
  (clean shaker473)
  (clean shot170)
  (clean shot283)
  (empty shaker473)
  (empty shot170)
  (empty shot283)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker473 l0)
  (shaker-level shaker473 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient10)
  (cocktail-part2 cocktail1 ingredient133)
)
 (:goal
  (and
      (contains shot170 cocktail1)
)))
