(define (problem prob)
 (:domain barman)
 (:objects 
      shaker128 - shaker
      left right - hand
      shot308 shot328 shot69 - shot
      ingredient455 ingredient39 ingredient412 ingredient341 - ingredient
      cocktail51 - cocktail
      dispenser242 dispenser339 dispenser5 dispenser17 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker128)
  (ontable shot308)
  (ontable shot328)
  (ontable shot69)
  (dispenses dispenser242 ingredient455)
  (dispenses dispenser339 ingredient39)
  (dispenses dispenser5 ingredient412)
  (dispenses dispenser17 ingredient341)
  (clean shaker128)
  (clean shot308)
  (clean shot328)
  (clean shot69)
  (empty shaker128)
  (empty shot308)
  (empty shot328)
  (empty shot69)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker128 l0)
  (shaker-level shaker128 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail51 ingredient455)
  (cocktail-part2 cocktail51 ingredient39)
)
 (:goal
  (and
      (contains shot308 cocktail51)
      (contains shot328 cocktail51)
)))
