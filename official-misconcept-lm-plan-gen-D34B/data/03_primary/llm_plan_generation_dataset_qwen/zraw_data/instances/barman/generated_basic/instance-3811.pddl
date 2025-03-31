(define (problem prob)
 (:domain barman)
 (:objects 
      shaker67 - shaker
      left right - hand
      shot36 shot60 - shot
      ingredient154 ingredient107 ingredient15 - ingredient
      cocktail202 - cocktail
      dispenser233 dispenser346 dispenser438 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker67)
  (ontable shot36)
  (ontable shot60)
  (dispenses dispenser233 ingredient154)
  (dispenses dispenser346 ingredient107)
  (dispenses dispenser438 ingredient15)
  (clean shaker67)
  (clean shot36)
  (clean shot60)
  (empty shaker67)
  (empty shot36)
  (empty shot60)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker67 l0)
  (shaker-level shaker67 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail202 ingredient154)
  (cocktail-part2 cocktail202 ingredient15)
)
 (:goal
  (and
      (contains shot36 cocktail202)
)))
