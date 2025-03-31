(define (problem prob)
 (:domain barman)
 (:objects 
      shaker218 - shaker
      left right - hand
      shot229 shot148 - shot
      ingredient497 ingredient270 - ingredient
      cocktail464 - cocktail
      dispenser484 dispenser139 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker218)
  (ontable shot229)
  (ontable shot148)
  (dispenses dispenser484 ingredient497)
  (dispenses dispenser139 ingredient270)
  (clean shaker218)
  (clean shot229)
  (clean shot148)
  (empty shaker218)
  (empty shot229)
  (empty shot148)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker218 l0)
  (shaker-level shaker218 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail464 ingredient497)
  (cocktail-part2 cocktail464 ingredient270)
)
 (:goal
  (and
      (contains shot229 cocktail464)
)))
