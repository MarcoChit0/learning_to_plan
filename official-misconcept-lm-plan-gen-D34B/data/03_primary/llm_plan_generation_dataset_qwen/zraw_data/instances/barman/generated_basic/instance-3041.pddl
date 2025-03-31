(define (problem prob)
 (:domain barman)
 (:objects 
      shaker345 - shaker
      left right - hand
      shot330 shot377 - shot
      ingredient414 ingredient409 ingredient422 - ingredient
      cocktail447 - cocktail
      dispenser426 dispenser485 dispenser353 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker345)
  (ontable shot330)
  (ontable shot377)
  (dispenses dispenser426 ingredient414)
  (dispenses dispenser485 ingredient409)
  (dispenses dispenser353 ingredient422)
  (clean shaker345)
  (clean shot330)
  (clean shot377)
  (empty shaker345)
  (empty shot330)
  (empty shot377)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker345 l0)
  (shaker-level shaker345 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail447 ingredient409)
  (cocktail-part2 cocktail447 ingredient422)
)
 (:goal
  (and
      (contains shot330 cocktail447)
)))
