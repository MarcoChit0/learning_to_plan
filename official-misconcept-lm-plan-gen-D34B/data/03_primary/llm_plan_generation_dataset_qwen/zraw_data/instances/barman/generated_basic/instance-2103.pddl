(define (problem prob)
 (:domain barman)
 (:objects 
      shaker320 - shaker
      left right - hand
      shot21 shot176 shot353 - shot
      ingredient209 ingredient153 - ingredient
      cocktail101 - cocktail
      dispenser322 dispenser223 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker320)
  (ontable shot21)
  (ontable shot176)
  (ontable shot353)
  (dispenses dispenser322 ingredient209)
  (dispenses dispenser223 ingredient153)
  (clean shaker320)
  (clean shot21)
  (clean shot176)
  (clean shot353)
  (empty shaker320)
  (empty shot21)
  (empty shot176)
  (empty shot353)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker320 l0)
  (shaker-level shaker320 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail101 ingredient209)
  (cocktail-part2 cocktail101 ingredient153)
)
 (:goal
  (and
      (contains shot21 cocktail101)
      (contains shot176 cocktail101)
)))
