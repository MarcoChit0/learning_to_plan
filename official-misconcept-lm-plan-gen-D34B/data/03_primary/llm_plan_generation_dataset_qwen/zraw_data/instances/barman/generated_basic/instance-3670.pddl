(define (problem prob)
 (:domain barman)
 (:objects 
      shaker317 - shaker
      left right - hand
      shot100 shot65 shot358 - shot
      ingredient453 ingredient75 ingredient415 ingredient250 - ingredient
      cocktail8 - cocktail
      dispenser333 dispenser63 dispenser477 dispenser77 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker317)
  (ontable shot100)
  (ontable shot65)
  (ontable shot358)
  (dispenses dispenser333 ingredient453)
  (dispenses dispenser63 ingredient75)
  (dispenses dispenser477 ingredient415)
  (dispenses dispenser77 ingredient250)
  (clean shaker317)
  (clean shot100)
  (clean shot65)
  (clean shot358)
  (empty shaker317)
  (empty shot100)
  (empty shot65)
  (empty shot358)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker317 l0)
  (shaker-level shaker317 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail8 ingredient75)
  (cocktail-part2 cocktail8 ingredient250)
)
 (:goal
  (and
      (contains shot100 cocktail8)
      (contains shot65 cocktail8)
)))
