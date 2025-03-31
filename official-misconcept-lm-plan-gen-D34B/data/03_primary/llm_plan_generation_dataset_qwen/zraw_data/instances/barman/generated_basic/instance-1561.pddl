(define (problem prob)
 (:domain barman)
 (:objects 
      shaker184 - shaker
      left right - hand
      shot77 shot465 shot320 - shot
      ingredient160 ingredient438 ingredient263 ingredient91 - ingredient
      cocktail97 - cocktail
      dispenser325 dispenser40 dispenser30 dispenser82 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker184)
  (ontable shot77)
  (ontable shot465)
  (ontable shot320)
  (dispenses dispenser325 ingredient160)
  (dispenses dispenser40 ingredient438)
  (dispenses dispenser30 ingredient263)
  (dispenses dispenser82 ingredient91)
  (clean shaker184)
  (clean shot77)
  (clean shot465)
  (clean shot320)
  (empty shaker184)
  (empty shot77)
  (empty shot465)
  (empty shot320)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker184 l0)
  (shaker-level shaker184 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail97 ingredient263)
  (cocktail-part2 cocktail97 ingredient160)
)
 (:goal
  (and
      (contains shot77 cocktail97)
      (contains shot465 cocktail97)
)))
