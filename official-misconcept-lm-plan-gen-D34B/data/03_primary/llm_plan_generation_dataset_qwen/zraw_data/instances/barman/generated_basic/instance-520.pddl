(define (problem prob)
 (:domain barman)
 (:objects 
      shaker87 - shaker
      left right - hand
      shot115 shot161 - shot
      ingredient252 ingredient71 - ingredient
      cocktail1 - cocktail
      dispenser493 dispenser176 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker87)
  (ontable shot115)
  (ontable shot161)
  (dispenses dispenser493 ingredient252)
  (dispenses dispenser176 ingredient71)
  (clean shaker87)
  (clean shot115)
  (clean shot161)
  (empty shaker87)
  (empty shot115)
  (empty shot161)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker87 l0)
  (shaker-level shaker87 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient252)
  (cocktail-part2 cocktail1 ingredient71)
)
 (:goal
  (and
      (contains shot115 cocktail1)
)))
