(define (problem prob)
 (:domain barman)
 (:objects 
      shaker278 - shaker
      left right - hand
      shot76 shot215 shot346 - shot
      ingredient250 ingredient55 ingredient426 - ingredient
      cocktail422 - cocktail
      dispenser254 dispenser358 dispenser173 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker278)
  (ontable shot76)
  (ontable shot215)
  (ontable shot346)
  (dispenses dispenser254 ingredient250)
  (dispenses dispenser358 ingredient55)
  (dispenses dispenser173 ingredient426)
  (clean shaker278)
  (clean shot76)
  (clean shot215)
  (clean shot346)
  (empty shaker278)
  (empty shot76)
  (empty shot215)
  (empty shot346)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker278 l0)
  (shaker-level shaker278 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail422 ingredient250)
  (cocktail-part2 cocktail422 ingredient55)
)
 (:goal
  (and
      (contains shot76 cocktail422)
      (contains shot215 ingredient55)
)))
