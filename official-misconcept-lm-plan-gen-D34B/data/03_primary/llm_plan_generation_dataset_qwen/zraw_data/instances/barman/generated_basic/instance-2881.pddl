(define (problem prob)
 (:domain barman)
 (:objects 
      shaker192 - shaker
      left right - hand
      shot452 shot81 shot151 - shot
      ingredient71 ingredient188 - ingredient
      cocktail440 - cocktail
      dispenser40 dispenser375 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker192)
  (ontable shot452)
  (ontable shot81)
  (ontable shot151)
  (dispenses dispenser40 ingredient71)
  (dispenses dispenser375 ingredient188)
  (clean shaker192)
  (clean shot452)
  (clean shot81)
  (clean shot151)
  (empty shaker192)
  (empty shot452)
  (empty shot81)
  (empty shot151)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker192 l0)
  (shaker-level shaker192 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail440 ingredient71)
  (cocktail-part2 cocktail440 ingredient188)
)
 (:goal
  (and
      (contains shot452 cocktail440)
      (contains shot81 cocktail440)
)))
