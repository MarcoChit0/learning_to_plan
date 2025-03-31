(define (problem prob)
 (:domain barman)
 (:objects 
      shaker350 - shaker
      left right - hand
      shot292 shot477 shot240 - shot
      ingredient93 ingredient20 ingredient67 ingredient414 - ingredient
      cocktail1 - cocktail
      dispenser181 dispenser53 dispenser383 dispenser128 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker350)
  (ontable shot292)
  (ontable shot477)
  (ontable shot240)
  (dispenses dispenser181 ingredient93)
  (dispenses dispenser53 ingredient20)
  (dispenses dispenser383 ingredient67)
  (dispenses dispenser128 ingredient414)
  (clean shaker350)
  (clean shot292)
  (clean shot477)
  (clean shot240)
  (empty shaker350)
  (empty shot292)
  (empty shot477)
  (empty shot240)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker350 l0)
  (shaker-level shaker350 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient20)
  (cocktail-part2 cocktail1 ingredient67)
)
 (:goal
  (and
      (contains shot292 cocktail1)
      (contains shot477 cocktail1)
)))
