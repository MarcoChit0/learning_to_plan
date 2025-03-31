(define (problem prob)
 (:domain barman)
 (:objects 
      shaker170 - shaker
      left right - hand
      shot101 shot149 shot392 - shot
      ingredient127 ingredient460 ingredient339 ingredient499 - ingredient
      cocktail190 - cocktail
      dispenser473 dispenser46 dispenser61 dispenser281 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker170)
  (ontable shot101)
  (ontable shot149)
  (ontable shot392)
  (dispenses dispenser473 ingredient127)
  (dispenses dispenser46 ingredient460)
  (dispenses dispenser61 ingredient339)
  (dispenses dispenser281 ingredient499)
  (clean shaker170)
  (clean shot101)
  (clean shot149)
  (clean shot392)
  (empty shaker170)
  (empty shot101)
  (empty shot149)
  (empty shot392)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker170 l0)
  (shaker-level shaker170 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail190 ingredient499)
  (cocktail-part2 cocktail190 ingredient339)
)
 (:goal
  (and
      (contains shot101 cocktail190)
      (contains shot149 cocktail190)
)))
