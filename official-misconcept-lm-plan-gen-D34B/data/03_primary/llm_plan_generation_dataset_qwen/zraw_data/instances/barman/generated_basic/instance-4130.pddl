(define (problem prob)
 (:domain barman)
 (:objects 
      shaker384 - shaker
      left right - hand
      shot199 shot328 - shot
      ingredient120 ingredient51 ingredient203 - ingredient
      cocktail257 - cocktail
      dispenser165 dispenser205 dispenser461 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker384)
  (ontable shot199)
  (ontable shot328)
  (dispenses dispenser165 ingredient120)
  (dispenses dispenser205 ingredient51)
  (dispenses dispenser461 ingredient203)
  (clean shaker384)
  (clean shot199)
  (clean shot328)
  (empty shaker384)
  (empty shot199)
  (empty shot328)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker384 l0)
  (shaker-level shaker384 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail257 ingredient203)
  (cocktail-part2 cocktail257 ingredient120)
)
 (:goal
  (and
      (contains shot199 cocktail257)
)))
