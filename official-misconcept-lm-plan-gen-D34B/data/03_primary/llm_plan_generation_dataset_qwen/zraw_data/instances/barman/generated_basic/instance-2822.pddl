(define (problem prob)
 (:domain barman)
 (:objects 
      shaker241 - shaker
      left right - hand
      shot405 shot275 shot245 - shot
      ingredient462 ingredient321 ingredient259 ingredient373 - ingredient
      cocktail80 - cocktail
      dispenser163 dispenser196 dispenser225 dispenser103 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker241)
  (ontable shot405)
  (ontable shot275)
  (ontable shot245)
  (dispenses dispenser163 ingredient462)
  (dispenses dispenser196 ingredient321)
  (dispenses dispenser225 ingredient259)
  (dispenses dispenser103 ingredient373)
  (clean shaker241)
  (clean shot405)
  (clean shot275)
  (clean shot245)
  (empty shaker241)
  (empty shot405)
  (empty shot275)
  (empty shot245)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker241 l0)
  (shaker-level shaker241 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail80 ingredient373)
  (cocktail-part2 cocktail80 ingredient259)
)
 (:goal
  (and
      (contains shot405 cocktail80)
      (contains shot275 ingredient373)
)))
