(define (problem prob)
 (:domain barman)
 (:objects 
      shaker268 - shaker
      left right - hand
      shot123 shot148 - shot
      ingredient424 ingredient350 - ingredient
      cocktail31 - cocktail
      dispenser230 dispenser209 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker268)
  (ontable shot123)
  (ontable shot148)
  (dispenses dispenser230 ingredient424)
  (dispenses dispenser209 ingredient350)
  (clean shaker268)
  (clean shot123)
  (clean shot148)
  (empty shaker268)
  (empty shot123)
  (empty shot148)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker268 l0)
  (shaker-level shaker268 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail31 ingredient424)
  (cocktail-part2 cocktail31 ingredient350)
)
 (:goal
  (and
      (contains shot123 cocktail31)
)))
