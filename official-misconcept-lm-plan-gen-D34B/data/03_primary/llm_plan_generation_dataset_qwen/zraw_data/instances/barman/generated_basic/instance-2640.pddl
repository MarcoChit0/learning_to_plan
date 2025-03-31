(define (problem prob)
 (:domain barman)
 (:objects 
      shaker257 - shaker
      left right - hand
      shot344 shot410 - shot
      ingredient466 ingredient49 ingredient336 - ingredient
      cocktail427 - cocktail
      dispenser32 dispenser470 dispenser92 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker257)
  (ontable shot344)
  (ontable shot410)
  (dispenses dispenser32 ingredient466)
  (dispenses dispenser470 ingredient49)
  (dispenses dispenser92 ingredient336)
  (clean shaker257)
  (clean shot344)
  (clean shot410)
  (empty shaker257)
  (empty shot344)
  (empty shot410)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker257 l0)
  (shaker-level shaker257 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail427 ingredient49)
  (cocktail-part2 cocktail427 ingredient466)
)
 (:goal
  (and
      (contains shot344 cocktail427)
)))
