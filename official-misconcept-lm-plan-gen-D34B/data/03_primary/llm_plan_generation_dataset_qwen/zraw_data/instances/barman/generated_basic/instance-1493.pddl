(define (problem prob)
 (:domain barman)
 (:objects 
      shaker58 - shaker
      left right - hand
      shot460 shot255 - shot
      ingredient448 ingredient63 ingredient256 ingredient269 - ingredient
      cocktail292 - cocktail
      dispenser196 dispenser402 dispenser384 dispenser291 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker58)
  (ontable shot460)
  (ontable shot255)
  (dispenses dispenser196 ingredient448)
  (dispenses dispenser402 ingredient63)
  (dispenses dispenser384 ingredient256)
  (dispenses dispenser291 ingredient269)
  (clean shaker58)
  (clean shot460)
  (clean shot255)
  (empty shaker58)
  (empty shot460)
  (empty shot255)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker58 l0)
  (shaker-level shaker58 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail292 ingredient269)
  (cocktail-part2 cocktail292 ingredient63)
)
 (:goal
  (and
      (contains shot460 cocktail292)
)))
