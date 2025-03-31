(define (problem prob)
 (:domain barman)
 (:objects 
      shaker167 - shaker
      left right - hand
      shot118 shot350 shot497 - shot
      ingredient39 ingredient151 - ingredient
      cocktail233 - cocktail
      dispenser321 dispenser227 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker167)
  (ontable shot118)
  (ontable shot350)
  (ontable shot497)
  (dispenses dispenser321 ingredient39)
  (dispenses dispenser227 ingredient151)
  (clean shaker167)
  (clean shot118)
  (clean shot350)
  (clean shot497)
  (empty shaker167)
  (empty shot118)
  (empty shot350)
  (empty shot497)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker167 l0)
  (shaker-level shaker167 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail233 ingredient151)
  (cocktail-part2 cocktail233 ingredient39)
)
 (:goal
  (and
      (contains shot118 cocktail233)
      (contains shot350 cocktail233)
)))
