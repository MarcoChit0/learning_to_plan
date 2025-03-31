(define (problem prob)
 (:domain barman)
 (:objects 
      shaker162 - shaker
      left right - hand
      shot260 - shot
      ingredient19 ingredient391 ingredient238 ingredient268 - ingredient
      cocktail1 - cocktail
      dispenser458 dispenser479 dispenser469 dispenser264 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker162)
  (ontable shot260)
  (dispenses dispenser458 ingredient19)
  (dispenses dispenser479 ingredient391)
  (dispenses dispenser469 ingredient238)
  (dispenses dispenser264 ingredient268)
  (clean shaker162)
  (clean shot260)
  (empty shaker162)
  (empty shot260)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker162 l0)
  (shaker-level shaker162 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient19)
  (cocktail-part2 cocktail1 ingredient391)
)
 (:goal
  (and
      (contains shot260 cocktail1)
)))
