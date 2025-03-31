(define (problem prob)
 (:domain barman)
 (:objects 
      shaker79 - shaker
      left right - hand
      shot378 shot49 shot86 shot293 - shot
      ingredient134 ingredient410 ingredient333 ingredient9 - ingredient
      cocktail170 - cocktail
      dispenser235 dispenser368 dispenser56 dispenser12 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker79)
  (ontable shot378)
  (ontable shot49)
  (ontable shot86)
  (ontable shot293)
  (dispenses dispenser235 ingredient134)
  (dispenses dispenser368 ingredient410)
  (dispenses dispenser56 ingredient333)
  (dispenses dispenser12 ingredient9)
  (clean shaker79)
  (clean shot378)
  (clean shot49)
  (clean shot86)
  (clean shot293)
  (empty shaker79)
  (empty shot378)
  (empty shot49)
  (empty shot86)
  (empty shot293)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker79 l0)
  (shaker-level shaker79 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail170 ingredient333)
  (cocktail-part2 cocktail170 ingredient134)
)
 (:goal
  (and
      (contains shot378 cocktail170)
      (contains shot49 ingredient134)
      (contains shot86 cocktail170)
)))
