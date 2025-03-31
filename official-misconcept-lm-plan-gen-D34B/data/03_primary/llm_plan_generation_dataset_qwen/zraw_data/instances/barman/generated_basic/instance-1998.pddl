(define (problem prob)
 (:domain barman)
 (:objects 
      shaker298 - shaker
      left right - hand
      shot189 shot465 - shot
      ingredient111 ingredient21 ingredient330 - ingredient
      cocktail256 - cocktail
      dispenser256 dispenser328 dispenser308 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker298)
  (ontable shot189)
  (ontable shot465)
  (dispenses dispenser256 ingredient111)
  (dispenses dispenser328 ingredient21)
  (dispenses dispenser308 ingredient330)
  (clean shaker298)
  (clean shot189)
  (clean shot465)
  (empty shaker298)
  (empty shot189)
  (empty shot465)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker298 l0)
  (shaker-level shaker298 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail256 ingredient21)
  (cocktail-part2 cocktail256 ingredient330)
)
 (:goal
  (and
      (contains shot189 cocktail256)
)))
