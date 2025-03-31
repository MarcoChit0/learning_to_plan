(define (problem prob)
 (:domain barman)
 (:objects 
      shaker179 - shaker
      left right - hand
      shot377 shot21 shot85 - shot
      ingredient473 ingredient79 ingredient78 - ingredient
      cocktail160 - cocktail
      dispenser226 dispenser49 dispenser224 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker179)
  (ontable shot377)
  (ontable shot21)
  (ontable shot85)
  (dispenses dispenser226 ingredient473)
  (dispenses dispenser49 ingredient79)
  (dispenses dispenser224 ingredient78)
  (clean shaker179)
  (clean shot377)
  (clean shot21)
  (clean shot85)
  (empty shaker179)
  (empty shot377)
  (empty shot21)
  (empty shot85)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker179 l0)
  (shaker-level shaker179 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail160 ingredient79)
  (cocktail-part2 cocktail160 ingredient78)
)
 (:goal
  (and
      (contains shot377 cocktail160)
      (contains shot21 ingredient78)
)))
