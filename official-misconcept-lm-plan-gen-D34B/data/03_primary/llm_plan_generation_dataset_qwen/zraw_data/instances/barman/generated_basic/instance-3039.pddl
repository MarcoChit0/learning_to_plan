(define (problem prob)
 (:domain barman)
 (:objects 
      shaker398 - shaker
      left right - hand
      shot460 shot293 - shot
      ingredient203 ingredient430 ingredient316 - ingredient
      cocktail444 - cocktail
      dispenser250 dispenser173 dispenser148 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker398)
  (ontable shot460)
  (ontable shot293)
  (dispenses dispenser250 ingredient203)
  (dispenses dispenser173 ingredient430)
  (dispenses dispenser148 ingredient316)
  (clean shaker398)
  (clean shot460)
  (clean shot293)
  (empty shaker398)
  (empty shot460)
  (empty shot293)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker398 l0)
  (shaker-level shaker398 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail444 ingredient430)
  (cocktail-part2 cocktail444 ingredient316)
)
 (:goal
  (and
      (contains shot460 cocktail444)
)))
