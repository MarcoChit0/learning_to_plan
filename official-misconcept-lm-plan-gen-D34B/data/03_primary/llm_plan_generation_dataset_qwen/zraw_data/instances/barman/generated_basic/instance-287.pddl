(define (problem prob)
 (:domain barman)
 (:objects 
      shaker43 - shaker
      left right - hand
      shot360 shot408 - shot
      ingredient366 ingredient228 ingredient456 - ingredient
      cocktail1 - cocktail
      dispenser463 dispenser234 dispenser168 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker43)
  (ontable shot360)
  (ontable shot408)
  (dispenses dispenser463 ingredient366)
  (dispenses dispenser234 ingredient228)
  (dispenses dispenser168 ingredient456)
  (clean shaker43)
  (clean shot360)
  (clean shot408)
  (empty shaker43)
  (empty shot360)
  (empty shot408)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker43 l0)
  (shaker-level shaker43 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient228)
  (cocktail-part2 cocktail1 ingredient366)
)
 (:goal
  (and
      (contains shot360 cocktail1)
)))
