(define (problem prob)
 (:domain barman)
 (:objects 
      shaker51 - shaker
      left right - hand
      shot473 shot183 - shot
      ingredient226 ingredient313 ingredient44 ingredient372 - ingredient
      cocktail1 - cocktail
      dispenser169 dispenser192 dispenser15 dispenser400 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker51)
  (ontable shot473)
  (ontable shot183)
  (dispenses dispenser169 ingredient226)
  (dispenses dispenser192 ingredient313)
  (dispenses dispenser15 ingredient44)
  (dispenses dispenser400 ingredient372)
  (clean shaker51)
  (clean shot473)
  (clean shot183)
  (empty shaker51)
  (empty shot473)
  (empty shot183)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker51 l0)
  (shaker-level shaker51 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient44)
  (cocktail-part2 cocktail1 ingredient313)
)
 (:goal
  (and
      (contains shot473 cocktail1)
)))
