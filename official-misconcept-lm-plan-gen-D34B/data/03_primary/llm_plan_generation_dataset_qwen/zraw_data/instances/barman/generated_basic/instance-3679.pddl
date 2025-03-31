(define (problem prob)
 (:domain barman)
 (:objects 
      shaker403 - shaker
      left right - hand
      shot138 shot229 shot418 - shot
      ingredient366 ingredient323 ingredient138 ingredient372 - ingredient
      cocktail388 - cocktail
      dispenser138 dispenser260 dispenser85 dispenser84 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker403)
  (ontable shot138)
  (ontable shot229)
  (ontable shot418)
  (dispenses dispenser138 ingredient366)
  (dispenses dispenser260 ingredient323)
  (dispenses dispenser85 ingredient138)
  (dispenses dispenser84 ingredient372)
  (clean shaker403)
  (clean shot138)
  (clean shot229)
  (clean shot418)
  (empty shaker403)
  (empty shot138)
  (empty shot229)
  (empty shot418)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker403 l0)
  (shaker-level shaker403 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail388 ingredient138)
  (cocktail-part2 cocktail388 ingredient323)
)
 (:goal
  (and
      (contains shot138 cocktail388)
      (contains shot229 ingredient323)
)))
