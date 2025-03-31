(define (problem prob)
 (:domain barman)
 (:objects 
      shaker45 - shaker
      left right - hand
      shot72 shot124 - shot
      ingredient454 ingredient360 ingredient498 - ingredient
      cocktail304 - cocktail
      dispenser165 dispenser268 dispenser265 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker45)
  (ontable shot72)
  (ontable shot124)
  (dispenses dispenser165 ingredient454)
  (dispenses dispenser268 ingredient360)
  (dispenses dispenser265 ingredient498)
  (clean shaker45)
  (clean shot72)
  (clean shot124)
  (empty shaker45)
  (empty shot72)
  (empty shot124)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker45 l0)
  (shaker-level shaker45 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail304 ingredient498)
  (cocktail-part2 cocktail304 ingredient360)
)
 (:goal
  (and
      (contains shot72 cocktail304)
)))
