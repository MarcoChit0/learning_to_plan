(define (problem prob)
 (:domain barman)
 (:objects 
      shaker305 - shaker
      left right - hand
      shot215 shot425 shot462 - shot
      ingredient455 ingredient458 ingredient244 - ingredient
      cocktail418 - cocktail
      dispenser220 dispenser98 dispenser276 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker305)
  (ontable shot215)
  (ontable shot425)
  (ontable shot462)
  (dispenses dispenser220 ingredient455)
  (dispenses dispenser98 ingredient458)
  (dispenses dispenser276 ingredient244)
  (clean shaker305)
  (clean shot215)
  (clean shot425)
  (clean shot462)
  (empty shaker305)
  (empty shot215)
  (empty shot425)
  (empty shot462)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker305 l0)
  (shaker-level shaker305 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail418 ingredient455)
  (cocktail-part2 cocktail418 ingredient458)
)
 (:goal
  (and
      (contains shot215 cocktail418)
      (contains shot425 ingredient455)
)))
