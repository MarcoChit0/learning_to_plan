(define (problem prob)
 (:domain barman)
 (:objects 
      shaker306 - shaker
      left right - hand
      shot354 shot165 shot229 - shot
      ingredient217 ingredient181 ingredient168 - ingredient
      cocktail91 - cocktail
      dispenser386 dispenser486 dispenser299 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker306)
  (ontable shot354)
  (ontable shot165)
  (ontable shot229)
  (dispenses dispenser386 ingredient217)
  (dispenses dispenser486 ingredient181)
  (dispenses dispenser299 ingredient168)
  (clean shaker306)
  (clean shot354)
  (clean shot165)
  (clean shot229)
  (empty shaker306)
  (empty shot354)
  (empty shot165)
  (empty shot229)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker306 l0)
  (shaker-level shaker306 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail91 ingredient168)
  (cocktail-part2 cocktail91 ingredient217)
)
 (:goal
  (and
      (contains shot354 cocktail91)
      (contains shot165 ingredient168)
)))
