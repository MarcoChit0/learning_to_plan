(define (problem prob)
 (:domain barman)
 (:objects 
      shaker188 - shaker
      left right - hand
      shot268 - shot
      ingredient177 ingredient389 - ingredient
      cocktail45 - cocktail
      dispenser140 dispenser354 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker188)
  (ontable shot268)
  (dispenses dispenser140 ingredient177)
  (dispenses dispenser354 ingredient389)
  (clean shaker188)
  (clean shot268)
  (empty shaker188)
  (empty shot268)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker188 l0)
  (shaker-level shaker188 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail45 ingredient177)
  (cocktail-part2 cocktail45 ingredient389)
)
 (:goal
  (and
      (contains shot268 cocktail45)
)))
