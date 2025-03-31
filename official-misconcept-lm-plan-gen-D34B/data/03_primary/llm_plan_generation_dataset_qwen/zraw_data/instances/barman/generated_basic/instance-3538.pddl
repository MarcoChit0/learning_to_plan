(define (problem prob)
 (:domain barman)
 (:objects 
      shaker87 - shaker
      left right - hand
      shot472 - shot
      ingredient169 ingredient37 ingredient142 - ingredient
      cocktail404 - cocktail
      dispenser381 dispenser421 dispenser414 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker87)
  (ontable shot472)
  (dispenses dispenser381 ingredient169)
  (dispenses dispenser421 ingredient37)
  (dispenses dispenser414 ingredient142)
  (clean shaker87)
  (clean shot472)
  (empty shaker87)
  (empty shot472)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker87 l0)
  (shaker-level shaker87 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail404 ingredient142)
  (cocktail-part2 cocktail404 ingredient169)
)
 (:goal
  (and
      (contains shot472 cocktail404)
)))
