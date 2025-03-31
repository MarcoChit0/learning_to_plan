(define (problem prob)
 (:domain barman)
 (:objects 
      shaker387 - shaker
      left right - hand
      shot345 - shot
      ingredient84 ingredient491 - ingredient
      cocktail253 - cocktail
      dispenser211 dispenser443 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker387)
  (ontable shot345)
  (dispenses dispenser211 ingredient84)
  (dispenses dispenser443 ingredient491)
  (clean shaker387)
  (clean shot345)
  (empty shaker387)
  (empty shot345)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker387 l0)
  (shaker-level shaker387 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail253 ingredient491)
  (cocktail-part2 cocktail253 ingredient84)
)
 (:goal
  (and
      (contains shot345 cocktail253)
)))
