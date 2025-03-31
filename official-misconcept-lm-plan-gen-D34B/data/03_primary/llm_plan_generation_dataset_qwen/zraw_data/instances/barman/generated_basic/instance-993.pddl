(define (problem prob)
 (:domain barman)
 (:objects 
      shaker268 - shaker
      left right - hand
      shot49 shot245 - shot
      ingredient202 ingredient306 - ingredient
      cocktail1 - cocktail
      dispenser481 dispenser32 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker268)
  (ontable shot49)
  (ontable shot245)
  (dispenses dispenser481 ingredient202)
  (dispenses dispenser32 ingredient306)
  (clean shaker268)
  (clean shot49)
  (clean shot245)
  (empty shaker268)
  (empty shot49)
  (empty shot245)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker268 l0)
  (shaker-level shaker268 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient202)
  (cocktail-part2 cocktail1 ingredient306)
)
 (:goal
  (and
      (contains shot49 cocktail1)
)))
