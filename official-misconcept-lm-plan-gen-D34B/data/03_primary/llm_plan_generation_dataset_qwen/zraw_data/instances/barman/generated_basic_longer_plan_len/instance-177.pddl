(define (problem prob)
 (:domain barman)
 (:objects 
      shaker87 - shaker
      left right - hand
      shot123 shot369 shot469 shot45 - shot
      ingredient484 ingredient208 - ingredient
      cocktail1 - cocktail
      dispenser384 dispenser183 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker87)
  (ontable shot123)
  (ontable shot369)
  (ontable shot469)
  (ontable shot45)
  (dispenses dispenser384 ingredient484)
  (dispenses dispenser183 ingredient208)
  (clean shaker87)
  (clean shot123)
  (clean shot369)
  (clean shot469)
  (clean shot45)
  (empty shaker87)
  (empty shot123)
  (empty shot369)
  (empty shot469)
  (empty shot45)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker87 l0)
  (shaker-level shaker87 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient484)
  (cocktail-part2 cocktail1 ingredient208)
)
 (:goal
  (and
      (contains shot123 cocktail1)
      (contains shot369 ingredient484)
      (contains shot469 ingredient484)
)))
