(define (problem prob)
 (:domain barman)
 (:objects 
      shaker95 - shaker
      left right - hand
      shot206 shot110 shot230 - shot
      ingredient392 ingredient152 - ingredient
      cocktail418 - cocktail
      dispenser39 dispenser499 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker95)
  (ontable shot206)
  (ontable shot110)
  (ontable shot230)
  (dispenses dispenser39 ingredient392)
  (dispenses dispenser499 ingredient152)
  (clean shaker95)
  (clean shot206)
  (clean shot110)
  (clean shot230)
  (empty shaker95)
  (empty shot206)
  (empty shot110)
  (empty shot230)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker95 l0)
  (shaker-level shaker95 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail418 ingredient392)
  (cocktail-part2 cocktail418 ingredient152)
)
 (:goal
  (and
      (contains shot206 cocktail418)
      (contains shot110 cocktail418)
)))
