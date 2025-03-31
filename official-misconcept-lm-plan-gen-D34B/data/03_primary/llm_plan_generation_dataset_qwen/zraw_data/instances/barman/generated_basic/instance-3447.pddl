(define (problem prob)
 (:domain barman)
 (:objects 
      shaker17 - shaker
      left right - hand
      shot56 shot195 - shot
      ingredient461 ingredient338 ingredient17 ingredient32 - ingredient
      cocktail362 - cocktail
      dispenser47 dispenser269 dispenser64 dispenser487 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker17)
  (ontable shot56)
  (ontable shot195)
  (dispenses dispenser47 ingredient461)
  (dispenses dispenser269 ingredient338)
  (dispenses dispenser64 ingredient17)
  (dispenses dispenser487 ingredient32)
  (clean shaker17)
  (clean shot56)
  (clean shot195)
  (empty shaker17)
  (empty shot56)
  (empty shot195)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker17 l0)
  (shaker-level shaker17 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail362 ingredient461)
  (cocktail-part2 cocktail362 ingredient32)
)
 (:goal
  (and
      (contains shot56 cocktail362)
)))
