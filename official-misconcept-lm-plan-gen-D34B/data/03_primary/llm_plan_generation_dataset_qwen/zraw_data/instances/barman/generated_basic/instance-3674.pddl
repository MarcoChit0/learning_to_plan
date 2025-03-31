(define (problem prob)
 (:domain barman)
 (:objects 
      shaker444 - shaker
      left right - hand
      shot312 shot181 shot65 - shot
      ingredient312 ingredient318 ingredient352 ingredient82 - ingredient
      cocktail119 - cocktail
      dispenser386 dispenser274 dispenser81 dispenser112 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker444)
  (ontable shot312)
  (ontable shot181)
  (ontable shot65)
  (dispenses dispenser386 ingredient312)
  (dispenses dispenser274 ingredient318)
  (dispenses dispenser81 ingredient352)
  (dispenses dispenser112 ingredient82)
  (clean shaker444)
  (clean shot312)
  (clean shot181)
  (clean shot65)
  (empty shaker444)
  (empty shot312)
  (empty shot181)
  (empty shot65)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker444 l0)
  (shaker-level shaker444 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail119 ingredient352)
  (cocktail-part2 cocktail119 ingredient312)
)
 (:goal
  (and
      (contains shot312 cocktail119)
      (contains shot181 cocktail119)
)))
