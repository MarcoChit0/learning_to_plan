(define (problem prob)
 (:domain barman)
 (:objects 
      shaker245 - shaker
      left right - hand
      shot460 shot291 shot409 shot47 - shot
      ingredient476 ingredient236 ingredient104 - ingredient
      cocktail220 - cocktail
      dispenser416 dispenser179 dispenser193 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker245)
  (ontable shot460)
  (ontable shot291)
  (ontable shot409)
  (ontable shot47)
  (dispenses dispenser416 ingredient476)
  (dispenses dispenser179 ingredient236)
  (dispenses dispenser193 ingredient104)
  (clean shaker245)
  (clean shot460)
  (clean shot291)
  (clean shot409)
  (clean shot47)
  (empty shaker245)
  (empty shot460)
  (empty shot291)
  (empty shot409)
  (empty shot47)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker245 l0)
  (shaker-level shaker245 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail220 ingredient104)
  (cocktail-part2 cocktail220 ingredient236)
)
 (:goal
  (and
      (contains shot460 cocktail220)
      (contains shot291 ingredient236)
      (contains shot409 cocktail220)
)))
