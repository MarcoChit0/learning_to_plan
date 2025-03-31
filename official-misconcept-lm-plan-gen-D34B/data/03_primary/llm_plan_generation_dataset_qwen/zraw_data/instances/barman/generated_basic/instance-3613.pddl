(define (problem prob)
 (:domain barman)
 (:objects 
      shaker70 - shaker
      left right - hand
      shot255 shot332 shot94 - shot
      ingredient269 ingredient362 ingredient110 - ingredient
      cocktail407 - cocktail
      dispenser260 dispenser272 dispenser263 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker70)
  (ontable shot255)
  (ontable shot332)
  (ontable shot94)
  (dispenses dispenser260 ingredient269)
  (dispenses dispenser272 ingredient362)
  (dispenses dispenser263 ingredient110)
  (clean shaker70)
  (clean shot255)
  (clean shot332)
  (clean shot94)
  (empty shaker70)
  (empty shot255)
  (empty shot332)
  (empty shot94)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker70 l0)
  (shaker-level shaker70 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail407 ingredient110)
  (cocktail-part2 cocktail407 ingredient362)
)
 (:goal
  (and
      (contains shot255 cocktail407)
      (contains shot332 cocktail407)
)))
