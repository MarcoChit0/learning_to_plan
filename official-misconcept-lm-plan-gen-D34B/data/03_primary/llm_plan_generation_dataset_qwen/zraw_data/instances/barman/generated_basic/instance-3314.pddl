(define (problem prob)
 (:domain barman)
 (:objects 
      shaker293 - shaker
      left right - hand
      shot319 shot189 shot140 - shot
      ingredient279 ingredient392 ingredient202 - ingredient
      cocktail53 - cocktail
      dispenser32 dispenser354 dispenser314 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker293)
  (ontable shot319)
  (ontable shot189)
  (ontable shot140)
  (dispenses dispenser32 ingredient279)
  (dispenses dispenser354 ingredient392)
  (dispenses dispenser314 ingredient202)
  (clean shaker293)
  (clean shot319)
  (clean shot189)
  (clean shot140)
  (empty shaker293)
  (empty shot319)
  (empty shot189)
  (empty shot140)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker293 l0)
  (shaker-level shaker293 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail53 ingredient392)
  (cocktail-part2 cocktail53 ingredient202)
)
 (:goal
  (and
      (contains shot319 cocktail53)
      (contains shot189 cocktail53)
)))
