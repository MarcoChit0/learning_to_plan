(define (problem prob)
 (:domain barman)
 (:objects 
      shaker149 - shaker
      left right - hand
      shot92 shot135 shot390 - shot
      ingredient99 ingredient367 ingredient129 ingredient212 - ingredient
      cocktail271 - cocktail
      dispenser270 dispenser259 dispenser135 dispenser33 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker149)
  (ontable shot92)
  (ontable shot135)
  (ontable shot390)
  (dispenses dispenser270 ingredient99)
  (dispenses dispenser259 ingredient367)
  (dispenses dispenser135 ingredient129)
  (dispenses dispenser33 ingredient212)
  (clean shaker149)
  (clean shot92)
  (clean shot135)
  (clean shot390)
  (empty shaker149)
  (empty shot92)
  (empty shot135)
  (empty shot390)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker149 l0)
  (shaker-level shaker149 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail271 ingredient99)
  (cocktail-part2 cocktail271 ingredient129)
)
 (:goal
  (and
      (contains shot92 cocktail271)
      (contains shot135 ingredient99)
)))
