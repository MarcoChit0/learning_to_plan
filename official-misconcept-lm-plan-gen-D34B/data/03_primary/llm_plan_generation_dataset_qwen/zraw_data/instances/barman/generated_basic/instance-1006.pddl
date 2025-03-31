(define (problem prob)
 (:domain barman)
 (:objects 
      shaker415 - shaker
      left right - hand
      shot313 shot498 - shot
      ingredient268 ingredient247 - ingredient
      cocktail1 - cocktail
      dispenser351 dispenser186 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker415)
  (ontable shot313)
  (ontable shot498)
  (dispenses dispenser351 ingredient268)
  (dispenses dispenser186 ingredient247)
  (clean shaker415)
  (clean shot313)
  (clean shot498)
  (empty shaker415)
  (empty shot313)
  (empty shot498)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker415 l0)
  (shaker-level shaker415 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient268)
  (cocktail-part2 cocktail1 ingredient247)
)
 (:goal
  (and
      (contains shot313 cocktail1)
)))
