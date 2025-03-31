(define (problem prob)
 (:domain barman)
 (:objects 
      shaker375 - shaker
      left right - hand
      shot255 shot447 shot110 - shot
      ingredient474 ingredient95 ingredient370 - ingredient
      cocktail178 - cocktail
      dispenser211 dispenser17 dispenser334 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker375)
  (ontable shot255)
  (ontable shot447)
  (ontable shot110)
  (dispenses dispenser211 ingredient474)
  (dispenses dispenser17 ingredient95)
  (dispenses dispenser334 ingredient370)
  (clean shaker375)
  (clean shot255)
  (clean shot447)
  (clean shot110)
  (empty shaker375)
  (empty shot255)
  (empty shot447)
  (empty shot110)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker375 l0)
  (shaker-level shaker375 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail178 ingredient95)
  (cocktail-part2 cocktail178 ingredient474)
)
 (:goal
  (and
      (contains shot255 cocktail178)
      (contains shot447 ingredient370)
)))
