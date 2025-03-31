(define (problem prob)
 (:domain barman)
 (:objects 
      shaker226 - shaker
      left right - hand
      shot256 shot321 - shot
      ingredient126 ingredient373 ingredient143 ingredient437 - ingredient
      cocktail423 - cocktail
      dispenser275 dispenser395 dispenser377 dispenser143 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker226)
  (ontable shot256)
  (ontable shot321)
  (dispenses dispenser275 ingredient126)
  (dispenses dispenser395 ingredient373)
  (dispenses dispenser377 ingredient143)
  (dispenses dispenser143 ingredient437)
  (clean shaker226)
  (clean shot256)
  (clean shot321)
  (empty shaker226)
  (empty shot256)
  (empty shot321)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker226 l0)
  (shaker-level shaker226 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail423 ingredient373)
  (cocktail-part2 cocktail423 ingredient437)
)
 (:goal
  (and
      (contains shot256 cocktail423)
)))
