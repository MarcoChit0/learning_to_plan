(define (problem prob)
 (:domain barman)
 (:objects 
      shaker181 - shaker
      left right - hand
      shot398 shot406 shot146 - shot
      ingredient134 ingredient137 ingredient342 - ingredient
      cocktail207 - cocktail
      dispenser447 dispenser318 dispenser373 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker181)
  (ontable shot398)
  (ontable shot406)
  (ontable shot146)
  (dispenses dispenser447 ingredient134)
  (dispenses dispenser318 ingredient137)
  (dispenses dispenser373 ingredient342)
  (clean shaker181)
  (clean shot398)
  (clean shot406)
  (clean shot146)
  (empty shaker181)
  (empty shot398)
  (empty shot406)
  (empty shot146)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker181 l0)
  (shaker-level shaker181 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail207 ingredient134)
  (cocktail-part2 cocktail207 ingredient137)
)
 (:goal
  (and
      (contains shot398 cocktail207)
      (contains shot406 ingredient134)
)))
