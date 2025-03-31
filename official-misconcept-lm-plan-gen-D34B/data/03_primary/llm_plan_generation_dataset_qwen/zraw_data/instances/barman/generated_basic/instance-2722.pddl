(define (problem prob)
 (:domain barman)
 (:objects 
      shaker353 - shaker
      left right - hand
      shot377 shot218 shot405 - shot
      ingredient330 ingredient130 ingredient287 - ingredient
      cocktail21 - cocktail
      dispenser409 dispenser307 dispenser452 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker353)
  (ontable shot377)
  (ontable shot218)
  (ontable shot405)
  (dispenses dispenser409 ingredient330)
  (dispenses dispenser307 ingredient130)
  (dispenses dispenser452 ingredient287)
  (clean shaker353)
  (clean shot377)
  (clean shot218)
  (clean shot405)
  (empty shaker353)
  (empty shot377)
  (empty shot218)
  (empty shot405)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker353 l0)
  (shaker-level shaker353 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail21 ingredient130)
  (cocktail-part2 cocktail21 ingredient287)
)
 (:goal
  (and
      (contains shot377 cocktail21)
      (contains shot218 cocktail21)
)))
