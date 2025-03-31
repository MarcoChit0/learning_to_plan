(define (problem prob)
 (:domain barman)
 (:objects 
      shaker139 - shaker
      left right - hand
      shot213 shot188 - shot
      ingredient429 ingredient330 ingredient389 ingredient474 - ingredient
      cocktail281 - cocktail
      dispenser152 dispenser150 dispenser165 dispenser472 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker139)
  (ontable shot213)
  (ontable shot188)
  (dispenses dispenser152 ingredient429)
  (dispenses dispenser150 ingredient330)
  (dispenses dispenser165 ingredient389)
  (dispenses dispenser472 ingredient474)
  (clean shaker139)
  (clean shot213)
  (clean shot188)
  (empty shaker139)
  (empty shot213)
  (empty shot188)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker139 l0)
  (shaker-level shaker139 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail281 ingredient330)
  (cocktail-part2 cocktail281 ingredient389)
)
 (:goal
  (and
      (contains shot213 cocktail281)
)))
