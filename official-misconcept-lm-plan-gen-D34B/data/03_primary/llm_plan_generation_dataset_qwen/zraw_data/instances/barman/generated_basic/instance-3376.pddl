(define (problem prob)
 (:domain barman)
 (:objects 
      shaker71 - shaker
      left right - hand
      shot353 shot414 shot208 - shot
      ingredient86 ingredient280 ingredient21 - ingredient
      cocktail129 - cocktail
      dispenser146 dispenser447 dispenser215 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker71)
  (ontable shot353)
  (ontable shot414)
  (ontable shot208)
  (dispenses dispenser146 ingredient86)
  (dispenses dispenser447 ingredient280)
  (dispenses dispenser215 ingredient21)
  (clean shaker71)
  (clean shot353)
  (clean shot414)
  (clean shot208)
  (empty shaker71)
  (empty shot353)
  (empty shot414)
  (empty shot208)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker71 l0)
  (shaker-level shaker71 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail129 ingredient280)
  (cocktail-part2 cocktail129 ingredient21)
)
 (:goal
  (and
      (contains shot353 cocktail129)
      (contains shot414 ingredient280)
)))
