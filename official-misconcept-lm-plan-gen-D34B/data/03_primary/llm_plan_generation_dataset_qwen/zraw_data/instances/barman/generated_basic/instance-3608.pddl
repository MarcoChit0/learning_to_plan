(define (problem prob)
 (:domain barman)
 (:objects 
      shaker43 - shaker
      left right - hand
      shot436 shot49 shot156 - shot
      ingredient424 ingredient146 ingredient143 - ingredient
      cocktail33 - cocktail
      dispenser238 dispenser390 dispenser170 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker43)
  (ontable shot436)
  (ontable shot49)
  (ontable shot156)
  (dispenses dispenser238 ingredient424)
  (dispenses dispenser390 ingredient146)
  (dispenses dispenser170 ingredient143)
  (clean shaker43)
  (clean shot436)
  (clean shot49)
  (clean shot156)
  (empty shaker43)
  (empty shot436)
  (empty shot49)
  (empty shot156)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker43 l0)
  (shaker-level shaker43 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail33 ingredient143)
  (cocktail-part2 cocktail33 ingredient424)
)
 (:goal
  (and
      (contains shot436 cocktail33)
      (contains shot49 cocktail33)
)))
