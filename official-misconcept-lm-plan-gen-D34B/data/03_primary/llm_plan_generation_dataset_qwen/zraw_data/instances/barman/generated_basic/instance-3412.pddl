(define (problem prob)
 (:domain barman)
 (:objects 
      shaker323 - shaker
      left right - hand
      shot289 shot66 shot394 shot188 - shot
      ingredient92 ingredient110 ingredient236 - ingredient
      cocktail262 - cocktail
      dispenser320 dispenser166 dispenser89 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker323)
  (ontable shot289)
  (ontable shot66)
  (ontable shot394)
  (ontable shot188)
  (dispenses dispenser320 ingredient92)
  (dispenses dispenser166 ingredient110)
  (dispenses dispenser89 ingredient236)
  (clean shaker323)
  (clean shot289)
  (clean shot66)
  (clean shot394)
  (clean shot188)
  (empty shaker323)
  (empty shot289)
  (empty shot66)
  (empty shot394)
  (empty shot188)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker323 l0)
  (shaker-level shaker323 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail262 ingredient92)
  (cocktail-part2 cocktail262 ingredient236)
)
 (:goal
  (and
      (contains shot289 cocktail262)
      (contains shot66 cocktail262)
      (contains shot394 ingredient92)
)))
