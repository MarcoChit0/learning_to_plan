(define (problem prob)
 (:domain barman)
 (:objects 
      shaker107 - shaker
      left right - hand
      shot450 - shot
      ingredient391 ingredient381 ingredient271 ingredient358 - ingredient
      cocktail446 - cocktail
      dispenser328 dispenser284 dispenser0 dispenser165 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker107)
  (ontable shot450)
  (dispenses dispenser328 ingredient391)
  (dispenses dispenser284 ingredient381)
  (dispenses dispenser0 ingredient271)
  (dispenses dispenser165 ingredient358)
  (clean shaker107)
  (clean shot450)
  (empty shaker107)
  (empty shot450)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker107 l0)
  (shaker-level shaker107 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail446 ingredient271)
  (cocktail-part2 cocktail446 ingredient358)
)
 (:goal
  (and
      (contains shot450 cocktail446)
)))
