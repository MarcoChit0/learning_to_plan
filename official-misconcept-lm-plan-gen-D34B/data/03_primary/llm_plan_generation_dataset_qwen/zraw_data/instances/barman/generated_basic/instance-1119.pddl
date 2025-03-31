(define (problem prob)
 (:domain barman)
 (:objects 
      shaker243 - shaker
      left right - hand
      shot346 shot200 - shot
      ingredient101 ingredient288 - ingredient
      cocktail1 - cocktail
      dispenser188 dispenser279 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker243)
  (ontable shot346)
  (ontable shot200)
  (dispenses dispenser188 ingredient101)
  (dispenses dispenser279 ingredient288)
  (clean shaker243)
  (clean shot346)
  (clean shot200)
  (empty shaker243)
  (empty shot346)
  (empty shot200)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker243 l0)
  (shaker-level shaker243 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient288)
  (cocktail-part2 cocktail1 ingredient101)
)
 (:goal
  (and
      (contains shot346 cocktail1)
)))
