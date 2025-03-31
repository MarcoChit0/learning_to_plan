(define (problem prob)
 (:domain barman)
 (:objects 
      shaker370 - shaker
      left right - hand
      shot256 shot377 shot432 - shot
      ingredient59 ingredient438 - ingredient
      cocktail145 - cocktail
      dispenser24 dispenser96 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker370)
  (ontable shot256)
  (ontable shot377)
  (ontable shot432)
  (dispenses dispenser24 ingredient59)
  (dispenses dispenser96 ingredient438)
  (clean shaker370)
  (clean shot256)
  (clean shot377)
  (clean shot432)
  (empty shaker370)
  (empty shot256)
  (empty shot377)
  (empty shot432)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker370 l0)
  (shaker-level shaker370 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail145 ingredient438)
  (cocktail-part2 cocktail145 ingredient59)
)
 (:goal
  (and
      (contains shot256 cocktail145)
      (contains shot377 cocktail145)
)))
