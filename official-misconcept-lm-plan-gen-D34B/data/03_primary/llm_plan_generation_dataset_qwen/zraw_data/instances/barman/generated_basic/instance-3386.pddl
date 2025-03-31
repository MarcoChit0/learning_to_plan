(define (problem prob)
 (:domain barman)
 (:objects 
      shaker70 - shaker
      left right - hand
      shot377 shot158 shot179 - shot
      ingredient41 ingredient362 ingredient373 - ingredient
      cocktail404 - cocktail
      dispenser427 dispenser289 dispenser373 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker70)
  (ontable shot377)
  (ontable shot158)
  (ontable shot179)
  (dispenses dispenser427 ingredient41)
  (dispenses dispenser289 ingredient362)
  (dispenses dispenser373 ingredient373)
  (clean shaker70)
  (clean shot377)
  (clean shot158)
  (clean shot179)
  (empty shaker70)
  (empty shot377)
  (empty shot158)
  (empty shot179)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker70 l0)
  (shaker-level shaker70 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail404 ingredient373)
  (cocktail-part2 cocktail404 ingredient41)
)
 (:goal
  (and
      (contains shot377 cocktail404)
      (contains shot158 cocktail404)
)))
