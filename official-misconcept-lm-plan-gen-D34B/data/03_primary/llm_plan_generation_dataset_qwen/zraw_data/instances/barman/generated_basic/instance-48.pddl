(define (problem prob)
 (:domain barman)
 (:objects 
      shaker415 - shaker
      left right - hand
      shot201 shot124 - shot
      ingredient349 ingredient108 ingredient250 ingredient21 - ingredient
      cocktail1 - cocktail
      dispenser256 dispenser400 dispenser155 dispenser285 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker415)
  (ontable shot201)
  (ontable shot124)
  (dispenses dispenser256 ingredient349)
  (dispenses dispenser400 ingredient108)
  (dispenses dispenser155 ingredient250)
  (dispenses dispenser285 ingredient21)
  (clean shaker415)
  (clean shot201)
  (clean shot124)
  (empty shaker415)
  (empty shot201)
  (empty shot124)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker415 l0)
  (shaker-level shaker415 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient250)
  (cocktail-part2 cocktail1 ingredient21)
)
 (:goal
  (and
      (contains shot201 cocktail1)
)))
