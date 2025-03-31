(define (problem prob)
 (:domain barman)
 (:objects 
      shaker416 - shaker
      left right - hand
      shot398 shot186 - shot
      ingredient23 ingredient87 ingredient330 ingredient103 - ingredient
      cocktail87 - cocktail
      dispenser292 dispenser14 dispenser12 dispenser348 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker416)
  (ontable shot398)
  (ontable shot186)
  (dispenses dispenser292 ingredient23)
  (dispenses dispenser14 ingredient87)
  (dispenses dispenser12 ingredient330)
  (dispenses dispenser348 ingredient103)
  (clean shaker416)
  (clean shot398)
  (clean shot186)
  (empty shaker416)
  (empty shot398)
  (empty shot186)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker416 l0)
  (shaker-level shaker416 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail87 ingredient103)
  (cocktail-part2 cocktail87 ingredient330)
)
 (:goal
  (and
      (contains shot398 cocktail87)
)))
