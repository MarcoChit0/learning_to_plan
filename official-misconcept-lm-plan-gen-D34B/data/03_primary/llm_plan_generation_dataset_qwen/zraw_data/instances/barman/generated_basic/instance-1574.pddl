(define (problem prob)
 (:domain barman)
 (:objects 
      shaker193 - shaker
      left right - hand
      shot31 shot388 shot370 - shot
      ingredient345 ingredient113 ingredient432 ingredient184 - ingredient
      cocktail234 - cocktail
      dispenser184 dispenser165 dispenser398 dispenser193 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker193)
  (ontable shot31)
  (ontable shot388)
  (ontable shot370)
  (dispenses dispenser184 ingredient345)
  (dispenses dispenser165 ingredient113)
  (dispenses dispenser398 ingredient432)
  (dispenses dispenser193 ingredient184)
  (clean shaker193)
  (clean shot31)
  (clean shot388)
  (clean shot370)
  (empty shaker193)
  (empty shot31)
  (empty shot388)
  (empty shot370)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker193 l0)
  (shaker-level shaker193 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail234 ingredient113)
  (cocktail-part2 cocktail234 ingredient345)
)
 (:goal
  (and
      (contains shot31 cocktail234)
      (contains shot388 cocktail234)
)))
