(define (problem prob)
 (:domain barman)
 (:objects 
      shaker190 - shaker
      left right - hand
      shot475 shot104 shot447 - shot
      ingredient67 ingredient169 ingredient341 ingredient45 - ingredient
      cocktail418 - cocktail
      dispenser215 dispenser231 dispenser377 dispenser366 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker190)
  (ontable shot475)
  (ontable shot104)
  (ontable shot447)
  (dispenses dispenser215 ingredient67)
  (dispenses dispenser231 ingredient169)
  (dispenses dispenser377 ingredient341)
  (dispenses dispenser366 ingredient45)
  (clean shaker190)
  (clean shot475)
  (clean shot104)
  (clean shot447)
  (empty shaker190)
  (empty shot475)
  (empty shot104)
  (empty shot447)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker190 l0)
  (shaker-level shaker190 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail418 ingredient45)
  (cocktail-part2 cocktail418 ingredient341)
)
 (:goal
  (and
      (contains shot475 cocktail418)
      (contains shot104 cocktail418)
)))
