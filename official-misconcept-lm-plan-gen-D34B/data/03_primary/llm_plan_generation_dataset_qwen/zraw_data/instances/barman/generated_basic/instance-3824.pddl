(define (problem prob)
 (:domain barman)
 (:objects 
      shaker170 - shaker
      left right - hand
      shot470 shot15 - shot
      ingredient41 ingredient102 ingredient209 - ingredient
      cocktail220 - cocktail
      dispenser76 dispenser348 dispenser223 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker170)
  (ontable shot470)
  (ontable shot15)
  (dispenses dispenser76 ingredient41)
  (dispenses dispenser348 ingredient102)
  (dispenses dispenser223 ingredient209)
  (clean shaker170)
  (clean shot470)
  (clean shot15)
  (empty shaker170)
  (empty shot470)
  (empty shot15)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker170 l0)
  (shaker-level shaker170 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail220 ingredient102)
  (cocktail-part2 cocktail220 ingredient209)
)
 (:goal
  (and
      (contains shot470 cocktail220)
)))
