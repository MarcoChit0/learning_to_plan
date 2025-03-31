(define (problem prob)
 (:domain barman)
 (:objects 
      shaker26 - shaker
      left right - hand
      shot483 shot149 shot24 - shot
      ingredient268 ingredient15 - ingredient
      cocktail334 - cocktail
      dispenser79 dispenser323 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker26)
  (ontable shot483)
  (ontable shot149)
  (ontable shot24)
  (dispenses dispenser79 ingredient268)
  (dispenses dispenser323 ingredient15)
  (clean shaker26)
  (clean shot483)
  (clean shot149)
  (clean shot24)
  (empty shaker26)
  (empty shot483)
  (empty shot149)
  (empty shot24)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker26 l0)
  (shaker-level shaker26 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail334 ingredient268)
  (cocktail-part2 cocktail334 ingredient15)
)
 (:goal
  (and
      (contains shot483 cocktail334)
      (contains shot149 ingredient268)
)))
