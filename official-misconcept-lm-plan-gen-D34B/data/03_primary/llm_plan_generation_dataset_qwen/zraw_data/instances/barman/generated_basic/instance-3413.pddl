(define (problem prob)
 (:domain barman)
 (:objects 
      shaker378 - shaker
      left right - hand
      shot438 shot62 shot321 shot303 - shot
      ingredient161 ingredient227 ingredient150 - ingredient
      cocktail236 - cocktail
      dispenser228 dispenser239 dispenser36 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker378)
  (ontable shot438)
  (ontable shot62)
  (ontable shot321)
  (ontable shot303)
  (dispenses dispenser228 ingredient161)
  (dispenses dispenser239 ingredient227)
  (dispenses dispenser36 ingredient150)
  (clean shaker378)
  (clean shot438)
  (clean shot62)
  (clean shot321)
  (clean shot303)
  (empty shaker378)
  (empty shot438)
  (empty shot62)
  (empty shot321)
  (empty shot303)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker378 l0)
  (shaker-level shaker378 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail236 ingredient161)
  (cocktail-part2 cocktail236 ingredient150)
)
 (:goal
  (and
      (contains shot438 cocktail236)
      (contains shot62 ingredient150)
      (contains shot321 cocktail236)
)))
