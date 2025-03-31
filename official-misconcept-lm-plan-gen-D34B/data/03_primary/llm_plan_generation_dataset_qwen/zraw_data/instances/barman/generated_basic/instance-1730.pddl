(define (problem prob)
 (:domain barman)
 (:objects 
      shaker198 - shaker
      left right - hand
      shot28 shot353 - shot
      ingredient68 ingredient180 ingredient447 ingredient444 - ingredient
      cocktail386 - cocktail
      dispenser160 dispenser152 dispenser179 dispenser205 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker198)
  (ontable shot28)
  (ontable shot353)
  (dispenses dispenser160 ingredient68)
  (dispenses dispenser152 ingredient180)
  (dispenses dispenser179 ingredient447)
  (dispenses dispenser205 ingredient444)
  (clean shaker198)
  (clean shot28)
  (clean shot353)
  (empty shaker198)
  (empty shot28)
  (empty shot353)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker198 l0)
  (shaker-level shaker198 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail386 ingredient68)
  (cocktail-part2 cocktail386 ingredient180)
)
 (:goal
  (and
      (contains shot28 cocktail386)
)))
