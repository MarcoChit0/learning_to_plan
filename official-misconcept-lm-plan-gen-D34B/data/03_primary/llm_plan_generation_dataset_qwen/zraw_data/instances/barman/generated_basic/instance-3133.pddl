(define (problem prob)
 (:domain barman)
 (:objects 
      shaker156 - shaker
      left right - hand
      shot247 shot84 shot179 - shot
      ingredient22 ingredient229 - ingredient
      cocktail191 - cocktail
      dispenser474 dispenser465 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker156)
  (ontable shot247)
  (ontable shot84)
  (ontable shot179)
  (dispenses dispenser474 ingredient22)
  (dispenses dispenser465 ingredient229)
  (clean shaker156)
  (clean shot247)
  (clean shot84)
  (clean shot179)
  (empty shaker156)
  (empty shot247)
  (empty shot84)
  (empty shot179)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker156 l0)
  (shaker-level shaker156 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail191 ingredient22)
  (cocktail-part2 cocktail191 ingredient229)
)
 (:goal
  (and
      (contains shot247 cocktail191)
      (contains shot84 cocktail191)
)))
