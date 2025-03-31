(define (problem prob)
 (:domain barman)
 (:objects 
      shaker466 - shaker
      left right - hand
      shot288 shot126 shot348 shot279 - shot
      ingredient160 ingredient194 - ingredient
      cocktail326 - cocktail
      dispenser205 dispenser53 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker466)
  (ontable shot288)
  (ontable shot126)
  (ontable shot348)
  (ontable shot279)
  (dispenses dispenser205 ingredient160)
  (dispenses dispenser53 ingredient194)
  (clean shaker466)
  (clean shot288)
  (clean shot126)
  (clean shot348)
  (clean shot279)
  (empty shaker466)
  (empty shot288)
  (empty shot126)
  (empty shot348)
  (empty shot279)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker466 l0)
  (shaker-level shaker466 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail326 ingredient194)
  (cocktail-part2 cocktail326 ingredient160)
)
 (:goal
  (and
      (contains shot288 cocktail326)
      (contains shot126 ingredient194)
      (contains shot348 cocktail326)
)))
