(define (problem prob)
 (:domain barman)
 (:objects 
      shaker259 - shaker
      left right - hand
      shot183 shot63 - shot
      ingredient163 ingredient499 - ingredient
      cocktail1 - cocktail
      dispenser1 dispenser167 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker259)
  (ontable shot183)
  (ontable shot63)
  (dispenses dispenser1 ingredient163)
  (dispenses dispenser167 ingredient499)
  (clean shaker259)
  (clean shot183)
  (clean shot63)
  (empty shaker259)
  (empty shot183)
  (empty shot63)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker259 l0)
  (shaker-level shaker259 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient163)
  (cocktail-part2 cocktail1 ingredient499)
)
 (:goal
  (and
      (contains shot183 cocktail1)
)))
