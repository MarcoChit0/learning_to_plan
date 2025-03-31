(define (problem prob)
 (:domain barman)
 (:objects 
      shaker117 - shaker
      left right - hand
      shot463 shot435 shot344 shot105 - shot
      ingredient138 ingredient191 ingredient493 - ingredient
      cocktail402 - cocktail
      dispenser6 dispenser114 dispenser295 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker117)
  (ontable shot463)
  (ontable shot435)
  (ontable shot344)
  (ontable shot105)
  (dispenses dispenser6 ingredient138)
  (dispenses dispenser114 ingredient191)
  (dispenses dispenser295 ingredient493)
  (clean shaker117)
  (clean shot463)
  (clean shot435)
  (clean shot344)
  (clean shot105)
  (empty shaker117)
  (empty shot463)
  (empty shot435)
  (empty shot344)
  (empty shot105)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker117 l0)
  (shaker-level shaker117 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail402 ingredient191)
  (cocktail-part2 cocktail402 ingredient138)
)
 (:goal
  (and
      (contains shot463 cocktail402)
      (contains shot435 ingredient138)
      (contains shot344 cocktail402)
)))
