(define (problem prob)
 (:domain barman)
 (:objects 
      shaker259 - shaker
      left right - hand
      shot68 shot456 shot193 - shot
      ingredient308 ingredient418 ingredient297 ingredient117 - ingredient
      cocktail1 - cocktail
      dispenser462 dispenser131 dispenser86 dispenser136 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker259)
  (ontable shot68)
  (ontable shot456)
  (ontable shot193)
  (dispenses dispenser462 ingredient308)
  (dispenses dispenser131 ingredient418)
  (dispenses dispenser86 ingredient297)
  (dispenses dispenser136 ingredient117)
  (clean shaker259)
  (clean shot68)
  (clean shot456)
  (clean shot193)
  (empty shaker259)
  (empty shot68)
  (empty shot456)
  (empty shot193)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker259 l0)
  (shaker-level shaker259 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient418)
  (cocktail-part2 cocktail1 ingredient308)
)
 (:goal
  (and
      (contains shot68 cocktail1)
      (contains shot456 ingredient418)
)))
