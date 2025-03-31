(define (problem prob)
 (:domain barman)
 (:objects 
      shaker439 - shaker
      left right - hand
      shot246 shot203 - shot
      ingredient414 ingredient168 - ingredient
      cocktail224 - cocktail
      dispenser8 dispenser239 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker439)
  (ontable shot246)
  (ontable shot203)
  (dispenses dispenser8 ingredient414)
  (dispenses dispenser239 ingredient168)
  (clean shaker439)
  (clean shot246)
  (clean shot203)
  (empty shaker439)
  (empty shot246)
  (empty shot203)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker439 l0)
  (shaker-level shaker439 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail224 ingredient414)
  (cocktail-part2 cocktail224 ingredient168)
)
 (:goal
  (and
      (contains shot246 cocktail224)
)))
